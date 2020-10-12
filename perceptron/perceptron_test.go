package perceptron

import (
	"fmt"
	"testing"
)

func TestInitialization(t *testing.T) {
	p := NewPerceptron(0)
	t.Logf("%+v", p)
}

func TestPerceptronTrain(t *testing.T) {
	inputs := []Tokenizer{
		Vector{0, 0, 1}, Vector{1, 1, 1},
		Vector{1, 0, 1}, Vector{0, 1, 0},
	}
	outputs := []float64{0, 1, 1, 0}
	goPerceptron := NewPerceptron(len(inputs[0].Tokenize()))
	Train(inputs, outputs, goPerceptron, 10000)

	result1 := goPerceptron.Think(Vector{0, 1, 0})
	result2 := goPerceptron.Think(Vector{1, 0, 1})

	t.Logf("%f ~= 0\n%f ~= 1", result1, result2)

	if result1 > 0.1 {
		panic(fmt.Sprintf("%f should be 0", result1))
	}
	if result2 < 0.9 {
		panic(fmt.Sprintf("%f should be 1", result2))
	}
}

func TestHousingData(t *testing.T) {
	nextHouse := loadHousingData()
	goPerceptron := NewPerceptron(3)
	TrainIterator(nextHouse, goPerceptron, 10000)
	result := goPerceptron.Think(Vector{1203, 3, 11})
	shouldBe := float64(239500)
	if result != shouldBe {
		panic(fmt.Sprintf("%f should be %f", result, shouldBe))
	}
}
