package perceptron

import (
	"encoding/csv"
	"io"
	"os"
	"strconv"
)

func loadHousingData() func() (Tokenizer, float64) {
	return loadData("../data/housing-data.csv")
}

func loadData(path string) func() (Tokenizer, float64) {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	reader := csv.NewReader(file)
	_, err = reader.Read()
	if err != nil {
		panic(err)
	}
	return func() (Tokenizer, float64) {
		row, err := reader.Read()
		if err == io.EOF {
			return nil, 0
		}
		if err != nil {
			panic(err)
		}
		rowLength := len(row)
		vector := make(Vector, rowLength-1)
		var result float64
		for cellIndex, cell := range row {
			token, err := strconv.ParseFloat(cell, 64)
			if err != nil {
				token = 0
				//log.Fatalf("Could not parse cell with value %s", cell)
			}
			if cellIndex == rowLength-1 {
				result = token
			} else {
				vector[cellIndex] = token
			}
		}
		return vector, result
	}
}
