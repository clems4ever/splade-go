package main

import (
	"fmt"

	"github.com/clems4ever/splade-go/spladego"
)

func main() {
	model, err := spladego.NewModel()
	if err != nil {
		panic(err)
	}
	defer model.Close()

	queries := []string{"what causes aging fast"}
	documents := []string{
		"UV-A light, specifically, is what mainly causes tanning, skin aging, and cataracts, UV-B causes sunburn, skin aging and skin cancer, and UV-C is the strongest, and therefore most effective at killing microorganisms. Again – single words and multiple bullets.",
		"Answers from Ronald Petersen, M.D. Yes, Alzheimer's disease usually worsens slowly. But its speed of progression varies, depending on a person's genetic makeup, environmental factors, age at diagnosis and other medical conditions. Still, anyone diagnosed with Alzheimer's whose symptoms seem to be progressing quickly — or who experiences a sudden decline — should see his or her doctor.",
		"Bell's palsy and Extreme tiredness and Extreme fatigue (2 causes) Bell's palsy and Extreme tiredness and Hepatitis (2 causes) Bell's palsy and Extreme tiredness and Liver pain (2 causes) Bell's palsy and Extreme tiredness and Lymph node swelling in children (2 causes)",
	}

	queryEmbeddings, err := model.EncodeQuery(queries)
	if err != nil {
		panic(err)
	}

	documentEmbeddings, err := model.EncodeDocument(documents)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Query embeddings shape: [%d, %d]\n", len(queryEmbeddings), len(queryEmbeddings[0]))
	fmt.Printf("Document embeddings shape: [%d, %d]\n", len(documentEmbeddings), len(documentEmbeddings[0]))

	similarities := model.Similarity(queryEmbeddings, documentEmbeddings)
	fmt.Printf("Similarities: %v\n", similarities)
}
