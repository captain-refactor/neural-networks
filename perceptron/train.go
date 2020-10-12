package perceptron

func Train(inputs []Tokenizer, outputs []float64, machine Machine, epochs int) {
	for epochIndex := 0; epochIndex < epochs; epochIndex++ {
		for exampleIndex, val := range inputs {
			machine.LearnByExample(val, outputs[exampleIndex])
		}
	}
}

func TrainIterator(next func() (Tokenizer, float64), machine Machine, epochs int) {
	for epochIndex := 0; epochIndex < epochs; epochIndex++ {
		for {
			val, result := next()
			if val == nil {
				break
			}
			machine.LearnByExample(val, result)
		}
	}
}
