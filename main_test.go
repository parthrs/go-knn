package main

import (
	"reflect"
	"testing"
)

func TestFindMaxOccurence(t *testing.T) {
	data1 := []float64{1.0, 20.0, 21.0, 43.0, 42.0, 43.0}
	data2 := []float64{99.0, 99.0, 98.0, 100.0, 200.0, 100.0, 201.0, 100.0}

	if retVal := FindMaxOccurence(data1); retVal != 43.0 {
		t.Errorf("Expected %f, got %f\n", 43.0, retVal)
	}
	if retVal := FindMaxOccurence(data2); retVal != 100.0 {
		t.Errorf("Expected %f, got %f\n", 100.0, retVal)
	}
}

func TestStringLineToFloatSlice(t *testing.T) {
	data1 := "5.1,3.5,1.4,0.2,Iris-setosa"
	data2 := "7.0,3.2,4.7,1.4,Iris-versicolor"
	retVal, err := StringLineToFloatSlice(data1)
	if err != nil {
		t.Errorf("Failed parsing string (%v)\n", err)
	}
	if same := reflect.DeepEqual(retVal, []float64{5.1, 3.5, 1.4, 0.2, 1.0}); !same {
		t.Errorf("Expected %v, got %v\n", []float64{5.1, 3.5, 1.4, 0.2, 1.0}, retVal)
	}
	retVal, err = StringLineToFloatSlice(data2)
	if err != nil {
		t.Errorf("Failed parsing string (%v)\n", err)
	}
	if same := reflect.DeepEqual(retVal, []float64{7.0, 3.2, 4.7, 1.4, 2.0}); !same {
		t.Errorf("Expected %v, got %v\n", []float64{7.0, 3.2, 4.7, 1.4, 2.0}, retVal)
	}
}

func TestReducedSlice(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{92.0, 93.0, 94.0, 95.0, 96.0},
		{44.0, 45.0, 54.0, 2.0, 1.0},
	}
	expected := [][]float64{
		{92.0, 93.0, 94.0, 95.0, 96.0},
		{44.0, 45.0, 54.0, 2.0, 1.0},
	}
	retVal := ReducedSlice(data, 0)
	if same := reflect.DeepEqual(retVal, expected); !same {
		t.Errorf("Expected %v, got %v\n", expected, retVal)
	}

	expected = [][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{44.0, 45.0, 54.0, 2.0, 1.0},
	}
	retVal = ReducedSlice(data, 1)
	if same := reflect.DeepEqual(retVal, expected); !same {
		t.Errorf("Expected %v, got %v\n", expected, retVal)
	}
}

func TestSplitDatasetIntoKFolds(t *testing.T) {
	data := [][]float64{
		{5.1, 3.5, 1.4, 0.2, 1.0},
		{4.9, 3.0, 1.4, 0.2, 1.0},
		{7.0, 3.2, 4.7, 1.4, 2.0},
		{6.4, 3.2, 4.5, 1.5, 2.0},
		{7.7, 3.8, 6.7, 2.2, 3.0},
	}
	expectedTwoFold := [][][]float64{
		{
			{5.1, 3.5, 1.4, 0.2, 1.0},
			{7.0, 3.2, 4.7, 1.4, 2.0},
			{7.7, 3.8, 6.7, 2.2, 3.0},
		},
		{
			{4.9, 3.0, 1.4, 0.2, 1.0},
			{6.4, 3.2, 4.5, 1.5, 2.0},
		},
	}
	expectedThreeFold := [][][]float64{
		{
			{5.1, 3.5, 1.4, 0.2, 1.0},
			{6.4, 3.2, 4.5, 1.5, 2.0},
		},
		{
			{4.9, 3.0, 1.4, 0.2, 1.0},
			{7.7, 3.8, 6.7, 2.2, 3.0},
		},
		{
			{7.0, 3.2, 4.7, 1.4, 2.0},
		},
	}
	expectedFourFold := [][][]float64{
		{
			{5.1, 3.5, 1.4, 0.2, 1.0},
			{7.7, 3.8, 6.7, 2.2, 3.0},
		},
		{
			{4.9, 3.0, 1.4, 0.2, 1.0},
		},
		{
			{7.0, 3.2, 4.7, 1.4, 2.0},
		},
		{
			{6.4, 3.2, 4.5, 1.5, 2.0},
		},
	}
	expected := SplitDatasetIntoKFolds(data, 2)
	if same := reflect.DeepEqual(expected, expectedTwoFold); !same {
		t.Errorf("Expected %v, got %v\n", expectedTwoFold, expected)
	}
	expected = SplitDatasetIntoKFolds(data, 3)
	if same := reflect.DeepEqual(expected, expectedThreeFold); !same {
		t.Errorf("Expected %v, got %v\n", expectedThreeFold, expected)
	}
	expected = SplitDatasetIntoKFolds(data, 4)
	if same := reflect.DeepEqual(expected, expectedFourFold); !same {
		t.Errorf("Expected %v, got %v\n", expectedFourFold, expected)
	}
}

func TestAccuracyMetric(t *testing.T) {
	actual := []float64{0.0, 1.0, 1.0, 0.0, 1.0}
	predicted := []float64{0.0, 1.0, 1.0, 0.0, 1.0}

	if result := AccuracyMetric(actual, predicted); result != 100.0 {
		t.Errorf("Expected 100.0, got %f\n", result)
	}

	predicted = []float64{0.0, 1.0, 1.0, 0.0, 2.0}
	if result := AccuracyMetric(actual, predicted); result != (4.0/5.0)*100.0 {
		t.Errorf("Expected %f, got %f\n", (4.0/5.0)*100.0, result)
	}
}

func TestFlattenSlice(t *testing.T) {
	data := [][][]float64{
		{
			{5.1, 3.5, 1.4, 0.2, 1.0},
			{6.4, 3.2, 4.5, 1.5, 2.0},
		},
		{
			{4.9, 3.0, 1.4, 0.2, 1.0},
			{7.7, 3.8, 6.7, 2.2, 3.0},
		},
		{
			{7.0, 3.2, 4.7, 1.4, 2.0},
		},
	}

	expected := [][]float64{
		{5.1, 3.5, 1.4, 0.2, 1.0},
		{6.4, 3.2, 4.5, 1.5, 2.0},
		{4.9, 3.0, 1.4, 0.2, 1.0},
		{7.7, 3.8, 6.7, 2.2, 3.0},
		{7.0, 3.2, 4.7, 1.4, 2.0},
	}

	result := FlattenSlice(data)
	if ok := reflect.DeepEqual(expected, result); !ok {
		t.Errorf("Expected: %v, got: %v\n", expected, result)
	}
}

func TestEuclideanDistance(t *testing.T) {
	i := []float64{6.3, 2.5, 5.0, 1.9}
	j := []float64{4.4, 3.0, 1.3, 0.2}
	k := []float64{5.9, 3.2, 4.8, 1.8}

	expected := float64(4.521061822182927)
	result, _ := EuclideanDistance(i, j)

	if equal := reflect.DeepEqual(expected, result); !equal {
		t.Errorf("Expected %f, got %f", expected, result)
	}

	expected = float64(0.8366600265340756)
	result, _ = EuclideanDistance(i, k)

	if equal := reflect.DeepEqual(expected, result); !equal {
		t.Errorf("Expected %f, got %f", expected, result)
	}
}
