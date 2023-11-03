package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// euclideanDistance calculates the distance between a set of points in the
// Euclidean space.
// Formula: https://en.wikipedia.org/wiki/Euclidean_distance#Higher_dimensions
func euclideanDistance(r1, r2 []float64) (float64, error) {
	if len(r1) == 0 {
		return 0.0, fmt.Errorf("zero length vector")
	}

	distance := float64(0)
	// Step-1: Calculate squares of the differences between the points and sum them
	for i := 0; i <= len(r1)-1; i++ { // Skip last col
		distance += math.Pow((r1[i] - r2[i]), 2)
	}

	// Step-2: Calculate the square root of the summation
	distance = math.Sqrt(distance)

	return distance, nil
}

type Distance struct {
	DataRow  []float64
	Distance float64
}

// getNeighbors returns k closest neighbors for the data in row
// with data points in training
func getNeighbors(training [][]float64, row []float64, k int) [][]float64 {
	neighbors := []Distance{}
	for _, val := range training {
		dist, err := euclideanDistance(row, val)
		if err == nil {
			neighbors = append(neighbors, Distance{val, dist})
		} else {
			fmt.Println(err)
			os.Exit(2)
		}
	}

	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})

	retVal := [][]float64{}

	for i := 0; i < k; i++ {
		retVal = append(retVal, neighbors[i].DataRow)
	}
	return retVal
}

// FindMaxOccurence returns the value that occurs the most
// in a slice
func FindMaxOccurence(input []float64) float64 {
	countMap := map[float64]int{}
	for _, i := range input {
		countMap[i] += 1
	}
	maxOccurence := 0
	var maxKey float64
	for k, v := range countMap {
		if v > maxOccurence {
			maxKey = k
			maxOccurence = v
		}
	}
	return maxKey
}

// PredictClassification makes a prediction matching the row with
// the class of the max neighbors matched
func PredictClassification(training [][]float64, row []float64, k int) float64 {
	neighbors := getNeighbors(training, row, k)
	lastElem := len(training[0]) - 1
	classes := []float64{}
	for _, i := range neighbors {
		classes = append(classes, i[lastElem])
	}
	return FindMaxOccurence(classes)
}

// loadCsv reads in a csv file and converts it into a "dataset"
// i.e. nested slice of row values as slices
func loadCsv(filename string) (dataset [][]float64, err error) {
	f, err := os.Open(filename)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		convertedToFloat, err := StringLineToFloatSlice(scanner.Text())
		if err != nil {
			return nil, err
		}
		dataset = append(dataset, convertedToFloat)
	}
	return
}

func mapClassToFloat(class string) float64 {
	return map[string]float64{
		"Iris-virginica":  0.0,
		"Iris-setosa":     1.0,
		"Iris-versicolor": 2.0,
	}[class]
}

func mapFloatToClass(f float64) string {
	return map[float64]string{
		0.0: "Iris-virginica",
		1.0: "Iris-setosa",
		2.0: "Iris-versicolor",
	}[f]
}

// StringLineToFloatSlice converts a comma separated line into a slice
// of floats
func StringLineToFloatSlice(line string) (fSlice []float64, err error) {
	splitString := strings.Split(line, ",")
	lastElem := len(splitString) - 1
	for i := 0; i < lastElem; i++ {
		convertedFloat, err := strconv.ParseFloat(splitString[i], 64)
		if err != nil {
			return nil, err
		}
		fSlice = append(fSlice, convertedFloat)
	}
	fSlice = append(fSlice, mapClassToFloat(splitString[lastElem]))
	return
}

// ReducedSlice pops the element at index and returns the slice
func ReducedSlice(dataset [][]float64, index int) (reducedDataset [][]float64) {
	finalSize := len(dataset) - 1
	reducedDataset = make([][]float64, finalSize)
	elemCounter := 0
	for i := range dataset {
		if i == index {
			continue
		}
		reducedDataset[elemCounter] = make([]float64, len(dataset[elemCounter]))
		copy(reducedDataset[elemCounter], dataset[i])
		elemCounter++
	}
	return
}

// SplitDatasetIntoKFolds splits a slice of slices into groups of numFolds
// and returns the result
func SplitDatasetIntoKFolds(dataset [][]float64, numFolds int) (foldedDataset [][][]float64) {
	if numFolds > len(dataset) {
		return
	}
	foldedDataset = make([][][]float64, numFolds)
	foldCtr := 0

	for _, elem := range dataset {
		foldedDataset[foldCtr] = append(foldedDataset[foldCtr], elem)
		foldCtr = (foldCtr + 1) % numFolds
	}

	return
}

func KNearestNeighbors(train [][]float64, test [][]float64, numNeighbors int) (predictions []float64) {
	for _, t := range test {
		predictions = append(predictions, PredictClassification(train, t, numNeighbors))
	}
	return
}

func AccuracyMetric(actual, predicted []float64) float64 {
	correct := 0
	for i := range actual {
		if actual[i] == predicted[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(actual)) * float64(100)
}

func FlattenSlice(structuredSlice [][][]float64) (flattenedSlice [][]float64) {

	for _, nestedSlice := range structuredSlice {
		for _, elem := range nestedSlice {
			flattenedSlice = append(flattenedSlice, elem)
		}
	}
	return
}

func EvaluateAlgorithm(dataset [][]float64, numNeighbors int, numFolds int, algo func([][]float64, [][]float64, int) []float64) []float64 {
	scores := []float64{}
	folds := SplitDatasetIntoKFolds(dataset, numFolds)

	for index, testData := range folds {
		predictionStatus := []float64{}
		actual := []float64{}
		var foldsCopy [][][]float64
		_ = copy(foldsCopy, folds)
		trainingData := FlattenSlice(append(foldsCopy[:index], foldsCopy[index+1:]...))

		for _, test := range testData {
			predictionStatus = append(predictionStatus, PredictClassification(trainingData, test, numNeighbors))
			//fmt.Printf("Prediction: %f\n", predictionStatus)
			actual = append(actual, test[len(test)-1])
			//fmt.Printf("Actual: %f\n", actual)
		}
		scores = append(scores, AccuracyMetric(actual, predictionStatus))
	}
	return scores
}

func main() {
	// Think of this as each element being a vector in a 2-D space
	// hence two data points (3rd element is just class type)
	// sampleData := [][]float64{
	// 	{2.7810836, 2.550537003, 0},
	// 	{1.465489372, 2.362125076, 0},
	// 	{3.396561688, 4.400293529, 0},
	// 	{1.38807019, 1.850220317, 0},
	// 	{3.06407232, 3.005305973, 0},
	// 	{7.627531214, 2.759262235, 1},
	// 	{5.332441248, 2.088626775, 1},
	// 	{6.922596716, 1.77106367, 1},
	// 	{8.675418651, -0.242068655, 1},
	// 	{7.673756466, 3.508563011, 1},
	// }

	// r0 := sampleData[0]

	//  fmt.Println(r0)

	// Simply prints the euclidean distance between two vectors
	// for i := range sampleData {
	// 	dist, err := euclideanDistance(r0, sampleData[i])
	// 	if err == nil {
	// 		fmt.Println(dist)
	// 	} else {
	// 		fmt.Printf("Error: %v\n", err)
	// 	}
	// }

	// 1. Calculates the distance between the given data with
	// the rest of the data set
	// 2. Sorts the result
	// 3. Prints 3 of the top results of point 2.
	// neighbors := getNeighbors(sampleData, r0, 3)
	// for i := range neighbors {
	// 	fmt.Println(neighbors[i])
	// }
	// prediction := predictClassification(sampleData, r0, 3)
	// fmt.Printf("Expected %d, Got %d\n", int(sampleData[0][2]), int(prediction))
	// for _, i := range splitDatasetIntoKFolds(sampleData, 3) {
	// 	fmt.Println(i)
	// }

	// Load dataset from the csv file
	dataset, err := loadCsv("iris.csv")
	if err != nil {
		fmt.Println(err)
		os.Exit(2)
	}

	// Make a single prediction
	testData := []float64{4.5, 2.3, 1.3, 0.3}
	fmt.Println(mapFloatToClass(PredictClassification(dataset, testData, 5)))

	// Evaluate algorithm and score predictions
	fmt.Println(EvaluateAlgorithm(dataset, 5, 5, KNearestNeighbors))
}
