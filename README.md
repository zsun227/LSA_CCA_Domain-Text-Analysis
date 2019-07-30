# LSA_CCA_Domain-Text-Analysis

This is the implementation of combining LSA and CCA methods for analyzing domain documents. For each word, a new representation vector is learned by taking the average of universal word embedding (e.g. Word2Vec, Glove) and LSA vector.

****
	
|Author|Zhongkai Sun|
|---|---
|E-mail|zsun227@qwisc.edu
****
## Guide
* [Functions](Functions)
* [Prerequisites](Prerequisites)
* [Usage](Usage)
* [Parameters ](Parameters )
* [Outputs](Outputs)

### Functions

#### Given sufficient documents from one domain, this code can check:
	1) Most changed words (compared with the universal word embedding)	
	2) Most closed words to any target word in the domain
  
#### Given sufficient documents from two domains, this code can check:
  	1) Most changed words (compared between two domains' representations)
  	2) Most closed words to any target in each domain.

### Prerequisites

This software is implemented in Python3. You may need install the following packages:

	Numpy 
	scipy
	h5py
	nltk
	sklearn
	joblib
	genism

You may also need to download the Glove word embedding 
* [Glove Embedding](https://nlp.stanford.edu/projects/glove/) - Used as the universal word embedding.

### Usage

First, you need to put all of your text data (.txt format) to the "data" folder. 

If you only have one domain, just put all of the .txt files in the "data" folder.

If you have two domains, you may need to put each domain's texts in the "data/train_two/first" and "data/train_two/second". 

All of the parameters can be changed in the "constant.py"

After setting up your parameters, you can run the code using this one-line code:

```
python3 main.py
```

The output will be generated in the "outputs" folder. 


### Parameters 



### Outputs

"models" folder contains all of models generated during the training. \\
"vectors" folder contains all of the word embedding vectors generated during the training.\\
"words" folder contains generated vocabularies during the training.\\
"words_analysis" contains outputs of most changed words & cloest words to a target word.
```
Give an example
```

) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

