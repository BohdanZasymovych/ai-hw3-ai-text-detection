## Data Analysis

Dataset contains texts in English and German from different domains like arxiv papers, student essays, cnn news, etc. Each text has a lable indicating how it was written: by human, AI, or fussion of both.

Dataset is unbalanced, the share of the texts written by AI is higher then that by human or fussion but the difference is not too big (25% by human, 45% by AI, 31% - fussion). The class balance by domain stays mostly the same as overall but for the gutenberg and arxiv papers 84-92% of the samples contain AI label (figure 1). Also dataset is inbalanced with respect to languges and domains some like DE bundestag, EN cnn news have much more samples then DE/EN gutenberg. Despite the distribution of languges and domains differs for train and test the same unbalance is kept.

All domains, languages and classes contain texts larger than 512 XLM-R tokens. Most combinations have share (in 20000 random sample from train) of such texts >= 75%, and DE gutenberg, EN arxiv papers have 100% of the texts longer than 512 tokens for human and AI classes, what will matter for the encoder (table 1).

### Train - val split

For the train part I have choosen 50000 texts and for val ~27000. A plain random subsample of 50k keeps about 18% of every cell, so a cell with fewer than about 2,700 texts would have fewer than 500 examples in it. That size is the floor and for cells with less than 2700 smaples we would keep all their texts, for others just sample so there should be no cell with much less than 500 texts. The val is just a sample which keeps the same distribution as original dataset to keep the validation fair and representable. (table 2, figure 2)


## TF-IDF model

I calculated TF-IDF score for 1, 2 word grams and 3-5 character grams, only 200k most popular were kept, then all values were normalized. I tried 3 ways of getting feature vector from the input text during inference calculating TF-IDF for the characters, for the words and together (the feature vector in this case have 400k values). To make an inference from the feature vector i tried linear Support Vector Machine and Logistic regression, for both models i tuned inverse regularization term $C$.

### Results

Logistic regression showed better results in general top 4 models have it, the best was combination of words and chars for features top 2 models have it. The best model is Logistic reggression with word and chars and $C=3$, it scored 0.870 macro F1 on the val and 0.873 on the public leaderboard. (table 3)

The model is almost perfect in classifing AI text: 99% of the predicted AI is right and only ~1-2% were AI false positives and ~1% Ai false negatives. But the model perplexes human and fussion. (figure - 3) 

## Encoder

As encoder model i choose pretrained xlm-roberta-base. To handle long texts I were using first 128 tokens of the text and last 382 (+ 2 special start and end tokens). This specific numbers were choosen based on the paper [cite How to Fine-Tune BERT for Text Classification?]. I trained for 2 epochs with learnign rate 2*10^-5. (figure 4)

### Results

Model got macro f1 0.911 on val and 0.915 on public leaderboard (0.870/0.873 for baseline). As TF-IDF model it perfectly handles AI but perplexes human and fussion but less than TF-IDF ([put some numbers here to justify the face]). (table 4, figure 5).

## Experiments and final pipeline

### Chunking

The main issue of the encoder is that it only sees beggining and ending of the text and doesnt see middle at all. The test showed that encoder scores on the long texts (> 2000 tokens) is worse than on shorter ones (510-1000): macro f1 0.781 against 0.936. (figure 6)

To handle this I descided to make experiment with chunking: text is split at at most 8 chunks. At inference for each chunk prediction (3 logits) is made and saved. Then i tried several approaches of utilizing this to get the predictions. One way was taking mean of logits/probabilities or replacing the mean of fussion logit by its max (logic is that one rewritten part of the text is enough to make it fussion and mean can average it with parts written by human or AI) but all this worsen the results f1 dropped by ~0.01 in comparison to it of regular encoder. Another way to encorporate probabilities for the windows was to use predictions for all windows as input to some other stacker model, in this case logistic reggression. And this approach showed great improvement: 0.911 (regular encoder) vs 0.917 (stacker on logits) vs 0.938 (stacker on window features). (table 5, figure 7)

### Perplexity

Another experiment to try to improve score was to use open weight LM (Qwen2.5-0.5B), it for a given sequence of tokens predicts probability distribution from which the next is sampled ($p_\theta(t_i \mid t_1, \dots, t_{i-1})$). Then the surprise for some token is calculated as $s_i = -\log p_\theta(t_i \mid t_{<i})$. Then the surprise score is calculated as mean surprise over 128 token chunks. The idea is that for the text generated by AI for the model it will be either to predict the next token and it will be less "surprised" (figure 8 which shows the mean surprise for chunks of 3 randomle sampled text shows that this assumption is reasonable, the surprise for the AI text is clearly smaller). Then the 5 features which can be used as inputs for the stucker are built from the surprise scores: mean, std, spread, min, max surprises for the chunks. Unfortunally feeding this features to the stacker didnt improved any thing plain stucker of probabilities scored 0.917 and stacker of probabilities and surprise scores scored the same score, plain stacker on surprise was much worse (0.504).

### Larger model && more data and train epochs

Reasonable assumption that using larger xlm-roberta-large model with ~560M parameters instead of xlm-roberta-base ~270M parameters will improve the score. Also on the training plots for the roberta the second epoch was still improving so I tried to add 3 epoch, and I descided to increase amount of train data from 50k texts to 100k. All this experiments were made together since it is reasonable to think that none will worsen the results.

Additionally i retrained the TF-IDF model on 100k text too.

#### Results

The model showed good improvements in comparison to the encoder trained before (0.911 vs 0.938) and it perplexes human and fussion less now (figure 9). The third epoch didnt add much, two likelly were enough (figure 10).

TF-IDF model showed slight improvements too (0.869 vs 0.880), (table 6).


### Final model

For the final model I descided to use stacker model since it showed big improvements and to experiment with giving different features to it. As features i tried logits of the large roberta, TF-IDF class scores, window features from the base and large roberta, surprise features. I tried all combinations of them to find the best one. As stacker I used logistic regression. I needed to train the stacker on each configuration and then evaluate it. I used 5 fold CV for this, to choose the best model i calculated the score on out-of-fold predictions. The best model was with the staker which takes logits of the large roberta, windows features of the large and small model, TF-IDf scores (0.960 macro f1). Because the score with this procedure can be a bit more optimistic i also checked the score with choose-and-fit nested procedure which gave 0.959 macro f1 and the best model won on 4 out of 5 folds. (table  7)

After the best model were choosed i compared logistic regression and xgboost as stuckers. xgboost won a bit (0.9600 vs 0.9637).

### Results

This model was the best of all with the highest macro f1 score on both val and public leaderboard (0.9637, 0.9632). Confussion matrix has 96%, 100%, 94% on the diagonal, and the model only a bit perplexes human and AI (figure 11).

### Model comparison

The table in the appendix shows the comparison of all models: table 8


## Appendix

figure 1 - label share per domain
table 1 - texts above 512 tokens per domain and label
table 2 - cell sizes
figure 2 - cell sizes

table 3 - models and search
figure 3 - confussion matrix baseline

figure 4 - roberta training plot
table 4 - encoder metrics
figure 5 - encoder confussion matrix

figure 6 - long texts
table 5 - encoder and stucker table
figure 7 - chunking confussion matrix stucker

figure 8 - model surprise for sampled texts

figure 9 - confussion matrix large model
figure 10 - large model training
table 6 - large model comparison

table 7 - stacker input combinations
figure 11 - confussion matrix final model
table 8



## Citations