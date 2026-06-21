# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

I used a Random Forest Classifier from scikit-learn, with 100 trees and a max depth of 10. I set `random_state=42` so I get the same results every time I run it. This is part of a project where I built the whole pipeline;  cleaning the data, training the model, checking how well it works, and then putting it behind an API so it can take requests and send back predictions.

## Intended Use

This model guesses whether someone makes more or less than $50K a year, based on facts such as their age, job, and education. I built this just for a school project to learn how the whole process works, not for any real use. It shouldn't be used to make actual decisions about real people. 

## Training Data

I used the Census Income dataset from the UCI repository ( the "Adult" dataset). It has 32,561 rows from the 1994 census. I used 80% of it to train the model, and I made sure the split kept the same mix of `<=50K` and `>50K` people in both the training and test sets. The columns include things like age, workclass, education, marital status, occupation, race, sex, hours worked per week, and native country.

## Evaluation Data

The other 20% of the data (about 6,513 rows) is what I used to test the model. I used the same split method as above. I also made sure to use the same encoder I trained on the training data instead of making a new one, so the test data gets treated the exact same way.

## Metrics

I checked precision, recall, and F1 score. Overall, the model got 0.8058 precision, 0.5453 recall, and 0.6504 F1. When I broke it down by group, the numbers move around a lot. People with a Doctorate got a recall of 0.8571, but people who only went to 7th or 8th grade got a recall of 0.0000. Mostly because there just aren't many people in that group who make over $50K. Some of the really small groups (such as countries with only a few people) show strange numbers like 0.0 or 1.0, but that's just because there's so little data for those groups, not because the model is extremely precise.

## Ethical Considerations

This data has race, sex, and country built into it, and the model doesn't do equally well across all of those groups. For example, recall for women (0.4163) is a lot lower than for men (0.5692), so the model misses more women who actually make over $50K. Also, since this is 1994 data, it's showing patterns from the early 90s instead of modern day accuracies. If anyone tried to use this for something real, it could end up being unfair in ways tied to that old data.

## Caveats and Recommendations

This was just a learning project, so there's a lot I'd do differently if it required more accurate results. I only did one train-test split instead of cross-validation, I didn't really tune the model much, and a bunch of the slice numbers come from really small groups so they're not that trustworthy on their own. If I kept working on this, I'd want to try tuning it with GridSearchCV, try other models like gradient boosting, and use newer data if this was ever going to matter outside of class.