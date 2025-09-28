package splade

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed tokenizer.json
var embeddedTokenizer []byte

//go:embed splade_pooled.onnx
var onnxModel []byte

type Model struct {
	tk      tokenizer.Tokenizer
	session *ort.DynamicAdvancedSession

	runtimePath string
}

type ModelOption = func(*Model)

func WithRuntimePath(path string) ModelOption {
	return func(m *Model) {
		m.runtimePath = path
	}
}

func NewModel(opts ...ModelOption) (*Model, error) {
	model := new(Model)

	for _, opt := range opts {
		opt(model)
	}

	tk, err := pretrained.FromReader(
		bytes.NewBuffer(embeddedTokenizer))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	if model.runtimePath != "" {
		ort.SetSharedLibraryPath(model.runtimePath)
	} else {
		path, ok := os.LookupEnv("ONNXRUNTIME_LIB_PATH")
		if ok {
			ort.SetSharedLibraryPath(path)
		}
	}

	err = ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize onnx runtime: %w", err)
	}

	// Create a dynamic session that accepts tensors at runtime
	inputNames := []string{"input_ids", "attention_mask"}
	outputNames := []string{"sparse_embedding"}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(onnxModel, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Model{
		tk:      *tk,
		session: session,
	}, nil
}

func (m *Model) Close() error {
	if m.session != nil {
		m.session.Destroy()
	}
	err := ort.DestroyEnvironment()
	return err
}

func (m *Model) Compute(sentence string, addSpecialTokens bool) ([]float32, error) {
	results, err := m.ComputeBatch([]string{sentence}, addSpecialTokens)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return results[0], nil
}

func (m *Model) ComputeFromEncoding(encoding tokenizer.Encoding) ([]float32, error) {
	res, err := m.ComputeBatchFromEncodings([]tokenizer.Encoding{encoding})
	if err != nil {
		return nil, err
	}
	return res[0], nil
}

func (m *Model) ComputeBatch(sentences []string, addSpecialTokens bool) ([][]float32, error) {
	if len(sentences) == 0 {
		return nil, nil
	}

	inputBatch := []tokenizer.EncodeInput{}
	for _, s := range sentences {
		inputBatch = append(inputBatch, tokenizer.NewSingleEncodeInput(tokenizer.NewRawInputSequence(s)))
	}

	if len(inputBatch) == 0 {
		return nil, nil
	}
	encodings, err := m.tk.EncodeBatch(inputBatch, addSpecialTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize sentence: %w", err)
	}
	return m.ComputeBatchFromEncodings(encodings)
}

func (m *Model) ComputeBatchFromEncodings(encodings []tokenizer.Encoding) ([][]float32, error) {
	batchSize := len(encodings)

	// Find the maximum sequence length in the batch
	maxSeqLength := 0
	for _, encoding := range encodings {
		if len(encoding.Ids) > maxSeqLength {
			maxSeqLength = len(encoding.Ids)
		}
	}

	vocabSize := 30522 // SPLADE vocabulary size

	inputShape := ort.NewShape(int64(batchSize), int64(maxSeqLength))

	// Create input tensors dynamically based on the maximum sequence length
	inputIdsData := make([]int64, batchSize*maxSeqLength)
	attentionMaskData := make([]int64, batchSize*maxSeqLength)

	for b := range batchSize {
		for i := range maxSeqLength {
			if i < len(encodings[b].Ids) {
				inputIdsData[b*maxSeqLength+i] = int64(encodings[b].Ids[i])
				attentionMaskData[b*maxSeqLength+i] = int64(encodings[b].AttentionMask[i])
			} else {
				// Padding with 0 for input_ids and 0 for attention_mask
				inputIdsData[b*maxSeqLength+i] = 0
				attentionMaskData[b*maxSeqLength+i] = 0
			}
		}
	}

	inputIdsTensor, err := ort.NewTensor(inputShape, inputIdsData)
	if err != nil {
		return nil, fmt.Errorf("failed creating input_ids tensor: %w", err)
	}
	defer inputIdsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(inputShape, attentionMaskData)
	if err != nil {
		return nil, fmt.Errorf("failed creating attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	sentenceOutputShape := ort.NewShape(int64(batchSize), int64(vocabSize))
	outputTensor, err := ort.NewEmptyTensor[float32](sentenceOutputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create empty tensor: %w", err)
	}
	defer outputTensor.Destroy()

	inputTensors := []ort.Value{inputIdsTensor, attentionMaskTensor}
	outputTensors := []ort.Value{outputTensor}

	err = m.session.Run(inputTensors, outputTensors)
	if err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	flatOutput := outputTensor.GetData()

	expectedTotalSize := batchSize * vocabSize
	if len(flatOutput) != expectedTotalSize {
		return nil, fmt.Errorf("unexpected output tensor size: got %d elements, expected %d elements", len(flatOutput), expectedTotalSize)
	}

	results := make([][]float32, batchSize)
	for i := range batchSize {
		start := i * vocabSize
		end := start + vocabSize
		results[i] = make([]float32, vocabSize)
		copy(results[i], flatOutput[start:end])
	}

	return results, nil
}

// EncodeQuery encodes queries using SPLADE
func (m *Model) EncodeQuery(queries []string) ([][]float32, error) {
	return m.ComputeBatch(queries, true)
}

// EncodeDocument encodes documents using SPLADE
func (m *Model) EncodeDocument(documents []string) ([][]float32, error) {
	return m.ComputeBatch(documents, true)
}

// Similarity computes cosine similarity between query and document embeddings
func (m *Model) Similarity(queryEmbeddings [][]float32, docEmbeddings [][]float32) [][]float32 {
	numQueries := len(queryEmbeddings)
	numDocs := len(docEmbeddings)

	similarities := make([][]float32, numQueries)
	for i := range numQueries {
		similarities[i] = make([]float32, numDocs)
		for j := range numDocs {
			similarities[i][j] = dotProduct(queryEmbeddings[i], docEmbeddings[j])
		}
	}

	return similarities
}

// dotProduct computes dot product between two sparse vectors
func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range len(a) {
		sum += a[i] * b[i]
	}
	return sum
}
