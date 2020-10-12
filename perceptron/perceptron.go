package perceptron

import (
	"math"
	"math/rand"
	"time"
)

type Perceptron struct {
	inputWidth    int
	iteration     int64
	weights       []float64
	bias          float64
	booleanResult bool
}

func NewPerceptron(inputWidth int) *Perceptron {
	perceptron := Perceptron{
		inputWidth: inputWidth,
		iteration:  0,
	}
	rand.Seed(time.Now().UnixNano())
	perceptron.bias = 0.0
	perceptron.weights = make([]float64, inputWidth)
	for i := 0; i < inputWidth; i++ {
		perceptron.weights[i] = rand.Float64()
	}
	return &perceptron
}

func (p *Perceptron) Think(input Tokenizer) float64 { //Forward Propagation
	return sigmoid(dotProduct(p.weights, input.Tokenize()) + p.bias)
}

func (p *Perceptron) LearnByExample(input Tokenizer, result float64) {
	p.iteration++
	actualResult := p.Think(input)
	p.correct(input.Tokenize(), result, actualResult)
}

func (p *Perceptron) calculateGradientOfWeights(input []float64, shouldBe float64, is float64) []float64 { //Calculate Gradients of Weights
	s := calculateGradientOfBias(shouldBe, is)
	result := make([]float64, len(input))
	for i := 0; i < len(input); i++ {
		result[i] += s * input[i] / 2
	}
	return result
}

func calculateGradientOfBias(shouldBe float64, is float64) float64 { //Calculate Gradients of Bias
	return -(is - shouldBe) * is * (1 - is)
}

func (p *Perceptron) correctWeights(correction []float64) {
	p.weights = addVectors(p.weights, correction)
}

func (p *Perceptron) correct(input []float64, shouldBe float64, is float64) {
	biasCorrection := 0.0
	weightsCorrection := make([]float64, p.inputWidth)
	weightsCorrection = addVectors(weightsCorrection, p.calculateGradientOfWeights(input, shouldBe, is))
	p.correctWeights(weightsCorrection)
	biasCorrection = calculateGradientOfBias(shouldBe, is)
	p.bias += biasCorrection / 2
}

//Dot Product of Two []float64s of same size
//Puts slices on top of another and
func dotProduct(v1, v2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return dot
}

//Addition of Two []float64s of same size
func addVectors(v1, v2 []float64) []float64 {
	add := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

func scalarMatMul(s float64, mat []float64) []float64 { //Multiplication of a []float64 & Matrix
	result := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		result[i] += s * mat[i]
	}
	return result
}

func sigmoid(x float64) float64 { //sigmoid Activation
	return 1.0 / (1.0 + math.Exp(-x))
}
