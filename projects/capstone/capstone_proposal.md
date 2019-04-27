# Machine Learning Engineer Nanodegree
## Capstone Proposal
Sitansh Rajput

January 10th, 2019

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

Humpback whales are of particular interest to many marine bioligists around the world right now. After decades and centuries of being intenstly hunted for their meat and fat, humpback popluations have fallen dramatically and are having a hard time recovering due to climate change. Climate change has created changes in whale migration, movement, and hunting as the increase in ocean temperature has caused their prey to move as well. Being able to automatically detect whales from their tales can give environment scientists and marine coservationists a lot of insight as to how to best further protect the whale populations.

I took on this project because ...

Increasing accessibility and education is an important goal within the STEM fields, and the applications for a machine learning powered ASL alphabet detector are enorous. Not only could such a model be fitted to other languages, but could be fitted to whole words and symbols. This could lead to educational services and apps focused on ASL to any variety of users.  

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

Each image is a grayscale (0-255) with a size of 28x28 and each set is contained within an independent .csv file. With a training set of around 20 thousand images, and a test set of around 5 thousand images, we need an efficient solution to detect the letter formed by a certain hand shape (as is done by speakers of ASL).
 
### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

The dataset was provided by Kaggle as an ASL drop-in for the classic MNIST dataset. It contains a training and testing .csv. The training .csv associates a hand sign with its respective label. Each row within the .csv contains a unique image. The end result would be to create a model with 95%+ greater accuracy. The algorithm will be tested on the test.csv set provided for this express purpose. 

https://www.kaggle.com/datamunge/sign-language-mnist

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

https://www.kaggle.com/ranjeetjain3/deep-learning-using-sign-langugage

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
