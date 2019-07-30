# LSA_CCA_Domain-Text-Analysis

This is the implementation of combining LSA and CCA methods for analyzing domain documents. For each word, a new representation vector is learned by taking the average of universal word embedding (e.g. Word2Vec, Glove) and LSA vector.

## Functions

1. Given sufficient documents from one domain, this code can check:
  1) Most changed words (compared with the universal word embedding)
  1) Most closed words to any target word in the domain
  
2. Given sufficient documents from two domains, this code can check:
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

### Basic Usage

First, you need to put all of your text data (.txt format) to the "data" folder. 

If you only have one domain, just put all of the .txt files in the "data" folder.

If you have two domains, you may need to put each domain's texts in the "data/train_two/first" and "data/train_two/second". 

All of the parameters can be changed in the "constant.py"

After setting up your parameters, you can run the code using this one-line code:

```
python3 main.py
```

The output will be generated in the "outputs" folder. 


## How to choose parameters 



### Outputs

"models" folder contains all of models generated during the training. \\
"vectors" folder contains all of the word embedding vectors generated during the training.\\
"words" folder contains generated vocabularies during the training.\\
"words_analysis" contains outputs of most changed words & cloest words to a target word.
```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

