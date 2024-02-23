package main

/*
#cgo LDFLAGS: -L. -LfastText/build -lstdc++ -l fasttext
#cgo CXXFLAGS: -std=c++17 -I. -I fastText/src/
#include <stdlib.h>
#include "gofasttext.h"
*/
import "C"
import (
	"flag"
	"log"
	"unsafe"
)

type FastText struct {
	modelPath string
}

type Prediction struct {
	Label       string
	Probability float32
}

func NewFastText() *FastText {
	return &FastText{}
}

func (ft *FastText) LoadModel(modelPath string) {
	ft.modelPath = modelPath
	cModelPath := C.CString(ft.modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	C.fasttext_LoadModel(cModelPath)
}

func (ft *FastText) Predict(line string, k int, threshold float32) []Prediction {

	var cOutputSize C.int
	cLine := C.CString(line)
	defer C.free(unsafe.Pointer(cLine))

	cK := C.int(k)
	cThreshold := C.float(threshold)

	cPredictions := C.fasttext_Predict(cLine, cK, cThreshold, &cOutputSize)

	if cPredictions == nil {
		return []Prediction{}
	}

	defer C.free(unsafe.Pointer(cPredictions))

	cPredictionsSlice := (*[1 << 30]C.struct_Prediction)(unsafe.Pointer(cPredictions))[:cOutputSize:cOutputSize]

	predictions := make([]Prediction, 0)

	for _, p := range cPredictionsSlice {
		predictions = append(predictions, Prediction{Label: C.GoString(p.label), Probability: float32(p.score)})
		C.free(unsafe.Pointer(p.label))
	}

	return predictions
}

func (ft *FastText) Test(line string, k int, threshold float32) {
	cLine := C.CString(line)
	defer C.free(unsafe.Pointer(cLine))

	cK := C.int(k)
	cThreshold := C.float(threshold)

	C.fasttext_Test(cLine, cK, cThreshold)

}

func main() {

	var flagLine string
	var flagModelPath string
	var flagThreshold float64
	var flagK int

	flag.StringVar(&flagLine, "line", "", "line to predict labels for")
	flag.StringVar(&flagModelPath, "model", "", "path to the model")

	flag.IntVar(&flagK, "k", 1, "k")
	flag.Float64Var(&flagThreshold, "threshold", 0.01, "threshold")

	flag.Parse()

	if flagModelPath == "" {
		log.Fatalf("model is required")
	}

	if flagLine == "" {
		log.Fatalf("line is required")
	}

	ft := NewFastText()
	ft.LoadModel(flagModelPath)

	ft.Test(flagLine, flagK, float32(flagThreshold))

	predictions := ft.Predict(flagLine, flagK, float32(flagThreshold))

	for _, p := range predictions {
		log.Printf("Label: %s, Probability: %f", p.Label, p.Probability)
	}
}
