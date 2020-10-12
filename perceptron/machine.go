package perceptron

type Tokenizer interface {
	Tokenize() []float64
}

type Machine interface {
	Think(input Tokenizer) float64
	LearnByExample(input Tokenizer, result float64)
}
