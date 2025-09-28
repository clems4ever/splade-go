# SPLADE-Go

[![Build and Test](https://github.com/clems4ever/splade-go/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/clems4ever/splade-go/actions/workflows/build-and-test.yml)

A Go implementation of SPLADE (SParse Lexical AnD Expansion) model for efficient semantic search using sparse representations.

## Overview

SPLADE is a neural information retrieval model that learns sparse representations of text, combining the efficiency of sparse retrieval with the effectiveness of dense semantic models. This Go implementation provides a fast, memory-efficient way to encode queries and documents into sparse vectors and compute similarities.

## Features

- **Sparse Encoding**: Convert text into sparse vector representations (30,522 dimensions - vocabulary size)
- **Query and Document Encoding**: Separate methods for encoding queries and documents
- **Similarity Computation**: Built-in cosine similarity calculation between sparse vectors
- **ONNX Runtime Integration**: Uses ONNX for fast inference
- **Batch Processing**: Efficient batch processing with automatic padding
- **Embedded Models**: Pre-compiled with tokenizer and ONNX model for easy deployment

## Installation

```bash
go get github.com/clems4ever/splade-go
```

### Prerequisites

You need to have the ONNX Runtime library available on your system. Set the environment variable:

```bash
export ONNXRUNTIME_LIB_PATH=libonnxruntime.so
```

Or provide the path programmatically when creating a new model:

```go
model, err := splade.NewModel(splade.WithRuntimePath("/path/to/libonnxruntime.so"))
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/clems4ever/splade-go/splade"
)

func main() {
    // Initialize the model
    model, err := splade.NewModel()
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // Define queries and documents
    queries := []string{"what causes aging fast"}
    documents := []string{
        "UV-A light causes skin aging and cataracts",
        "Alzheimer's disease progression varies by genetics",
        "Bell's palsy causes extreme tiredness",
    }

    // Encode queries and documents
    queryEmbeddings, err := model.EncodeQuery(queries)
    if err != nil {
        panic(err)
    }

    documentEmbeddings, err := model.EncodeDocument(documents)
    if err != nil {
        panic(err)
    }

    // Compute similarities
    similarities := model.Similarity(queryEmbeddings, documentEmbeddings)
    
    fmt.Printf("Query embeddings shape: [%d, %d]\n", len(queryEmbeddings), len(queryEmbeddings[0]))
    fmt.Printf("Document embeddings shape: [%d, %d]\n", len(documentEmbeddings), len(documentEmbeddings[0]))
    fmt.Printf("Similarities: %v\n", similarities)
}
```

## Model Details

- **Base Model**: `naver/splade-cocondenser-ensembledistil`
- **Output Dimensions**: 30,522 (vocabulary size)
- **Architecture**: BERT-based with SPLADE pooling (log(1+ReLU) activation + max pooling)
- **Tokenizer**: Embedded BERT tokenizer
- **ONNX Opset**: Version 14

## Building from Source

The project includes scripts to export the ONNX model from the original Python implementation:

```bash
# Export ONNX models (requires Python environment with transformers)
cd baseline
python export_onnx.py

# This generates:
# - splade_raw.onnx: Raw masked language model logits
# - splade_pooled.onnx: Pooled SPLADE embeddings (used by Go implementation)
```

## Dependencies

- [sugarme/tokenizer](https://github.com/sugarme/tokenizer) - BERT tokenizer implementation
- [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) - Go bindings for ONNX Runtime

## Performance

SPLADE provides an excellent balance between efficiency and effectiveness:

- **Sparse Representations**: Only non-zero dimensions are meaningful, enabling efficient storage and computation
- **Fast Inference**: ONNX Runtime provides optimized execution
- **Memory Efficient**: Batch processing with dynamic padding
- **Scalable**: Suitable for large-scale retrieval systems

## Use Cases

- **Semantic Search**: Find relevant documents for natural language queries
- **Information Retrieval**: Build search engines with neural ranking
- **Document Similarity**: Compare semantic similarity between texts
- **Question Answering**: Retrieve relevant context for QA systems

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under Apache 2.0 - see the LICENSE.md file for details.

## Acknowledgments

- Original SPLADE paper and implementation by the NAVER team
- Hugging Face for the pre-trained models and transformers library
- ONNX Runtime for efficient inference capabilities