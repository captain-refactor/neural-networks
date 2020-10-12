package perceptron

import (
	"fmt"
	"testing"
)

func TestDot(t *testing.T) {
	r := dotProduct([]float64{1, 2, 3}, []float64{1, 2, 3})
	fmt.Printf("%f", r)
}

func TestVectorAddition(t *testing.T) {
	r := addVectors([]float64{1, 2, 3, 4}, []float64{1, 2, 3, 4})
	fmt.Printf("%f", r)
}

func TestMatMultiplication(t *testing.T) {
	r := scalarMatMul(5, []float64{1, 2, 3, 4})
	fmt.Printf("%f", r)
}
