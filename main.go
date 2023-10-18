package main

import (
	"fmt"
	"math"
	"sort"
)

// euclideanDistance calculates the distance between a set of points in the
// Euclidean space.
// Formula: https://en.wikipedia.org/wiki/Euclidean_distance#Higher_dimensions
func euclideanDistance(r1, r2 []float64) (float64, error) {
	if len(r1) != len(r2) {
		return 0.0, fmt.Errorf("vector length mismatch")
	}
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
func getNeighbors(training [][]float64, row []float64, k int32) [][]float64 {
	neighbors := []Distance{}
	for _, val := range training {
		dist, err := euclideanDistance(row, val)
		if err == nil {
			neighbors = append(neighbors, Distance{val, dist})
		}
	}

	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})

	retVal := [][]float64{}

	for i := int32(0); i < k; i++ {
		retVal = append(retVal, neighbors[i].DataRow)
	}
	return retVal
}

func main() {
	sampleData := [][]float64{
		{2.7810836, 2.550537003, 0},
		{1.465489372, 2.362125076, 0},
		{3.396561688, 4.400293529, 0},
		{1.38807019, 1.850220317, 0},
		{3.06407232, 3.005305973, 0},
		{7.627531214, 2.759262235, 1},
		{5.332441248, 2.088626775, 1},
		{6.922596716, 1.77106367, 1},
		{8.675418651, -0.242068655, 1},
		{7.673756466, 3.508563011, 1},
	}

	r0 := sampleData[0]
	//fmt.Println(r0)

	// for i := range sampleData {
	// 	dist, err := euclideanDistance(r0, sampleData[i])
	// 	if err == nil {
	// 		fmt.Println(dist)
	// 	} else {
	// 		fmt.Printf("Error: %v\n", err)
	// 	}
	// }

	neighbors := getNeighbors(sampleData, r0, 3)
	for i := range neighbors {
		fmt.Println(neighbors[i])
	}
}
