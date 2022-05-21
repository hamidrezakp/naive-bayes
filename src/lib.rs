use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

type Class = String;
type Word = String;

#[derive(Clone)]
pub struct Document {
    pub class: Class,
    pub text: String,
}

impl Document {
    pub fn words(&self) -> Vec<&str> {
        self.text.split_whitespace().collect()
    }
}

pub struct NaiveBayes {
    vocab: HashSet<Word>,
    classes: HashSet<Class>,
    log_prior: HashMap<Class, f64>,
    likelihood: HashMap<(Class, Word), f64>,
}

impl NaiveBayes {
    pub fn new(documents: &[Document], classes: HashSet<Class>, vocab: HashSet<Word>) -> Self {
        let (log_prior, likelihood) = classes
            .iter()
            .map(|class| {
                println!("# starting to train class {}", class);
                let class_documents: Vec<_> =
                    documents.iter().filter(|doc| doc.class == *class).collect();

                println!("# starting to collect all words");
                let class_documents_words: Vec<_> =
                    class_documents.iter().flat_map(|doc| doc.words()).collect();

                println!("# starting to count all words");
                let class_words_count: usize = vocab
                    .iter()
                    .map(|v| class_documents_words.iter().filter(|w| *w == v).count() + 1)
                    .sum();

                println!("# starting to log prior");
                let log_prior = ((documents.len() / class_documents.len()) as f64).log2();

                println!("# starting to log likelihood");
                let likelihood: HashMap<(Class, Word), f64> = vocab
                    .iter()
                    .map(|word| {
                        println!("# starting to train word {}", word);
                        let count = class_documents_words.iter().filter(|w| *w == word).count();
                        let likelihood = (((count + 1) / class_words_count) as f64).log2();
                        ((class.clone(), word.clone()), likelihood)
                    })
                    .collect();
                ((class.clone(), log_prior), likelihood)
            })
            .fold(
                (HashMap::new(), HashMap::new()),
                |(mut s_log_prior, mut s_likelihood), (log_prior, likelihood)| {
                    s_log_prior.insert(log_prior.0, log_prior.1);
                    s_likelihood.extend(likelihood);
                    (s_log_prior, s_likelihood)
                },
            );

        Self {
            vocab,
            classes,
            log_prior,
            likelihood,
        }
    }

    pub fn guess(&self, document: &Document) -> Vec<Class> {
        let mut sum = self.log_prior.clone();
        for class in self.classes.iter() {
            for word in document.words() {
                if self.vocab.contains(word) {
                    sum.insert(
                        class.to_string(),
                        sum[class.as_str()] + self.likelihood[&(class.clone(), word.to_string())],
                    );
                }
            }
        }

        let max_item = sum.iter().max_by(|(_, i), (_, j)| {
            if i < j {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });

        match max_item {
            None => Vec::new(),
            Some((_, max_value)) => self
                .log_prior
                .iter()
                .map_while(|(class, value)| {
                    if value == max_value {
                        Some(class.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }
}

mod tests {
    use super::*;
    use std::{
        fs::{read_dir, read_to_string},
        path::{Path, PathBuf},
    };

    struct Dataset {
        pub vocab: HashSet<Word>,
        pub classes: HashSet<Class>,
        pub train_docs: Vec<Document>,
        pub test_docs: Vec<Document>,
    }

    fn read_folder_documents(path: &Path) -> Vec<Document> {
        read_dir(path)
            .unwrap()
            .map(|item| match item {
                Ok(i) if i.path().is_dir() => {
                    println!("## reading folder: {:?}", i.file_name());
                    let class = i.file_name();
                    let docs: Vec<Document> = read_dir(i.path())
                        .unwrap()
                        .map_while(|file| match file {
                            Ok(file) if file.path().is_file() => {
                                println!("## reading file: {:?}", file.file_name());
                                let text = read_to_string(file.path()).unwrap();
                                Some(Document {
                                    class: class.to_str().unwrap().into(),
                                    text,
                                })
                            }
                            _ => None,
                        })
                        .collect();
                    docs
                }
                _ => [].to_vec(),
            })
            .fold(Vec::new(), |mut s, v| {
                s.extend(v);
                s
            })
    }

    fn read_dataset<'a>(path: &'a str) -> Result<Dataset, &'static str> {
        if !Path::new(path).is_dir() {
            return Err("path must be folder of dataset");
        }

        let vocab = read_to_string(PathBuf::from_iter([path, "imdb.vocab"])).unwrap();
        let vocab: HashSet<Word> = vocab.split_whitespace().map(|s| s.to_owned()).collect();

        let train_path = PathBuf::from_iter([path, "train"]);
        let train_docs = read_folder_documents(&train_path);
        let classes: HashSet<Class> = train_docs.iter().map(|d| d.class.to_owned()).collect();

        let test_path = PathBuf::from_iter([path, "test"]);
        let test_docs = read_folder_documents(&test_path);

        Ok(Dataset {
            vocab,
            classes,
            train_docs,
            test_docs,
        })
    }
    #[test]
    fn test_train() {
        println!("### starting to read dataset");
        let dataset = read_dataset("dataset").unwrap();
        println!("### dataset read successfully");

        println!("### starting to train");
        let naive_bayes = NaiveBayes::new(&dataset.train_docs, dataset.classes, dataset.vocab);
        println!("### naive_bayes made successfully");

        println!("### starting to guess");
        let guess = naive_bayes.guess(&dataset.test_docs[0]);
        println!("### guess: {:#?}", guess);
    }
}
