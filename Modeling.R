# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics
library(visdat)   # for additional visualizations
library(caret)    # for various ML tasks
library(e1071)
library(range)
library(rsample)
library(recipes)
library(klaR)
library(kernlab)
library(xgboost)
library(gbm)
library(gridExtra)

#------------------------- Revenue Classification -------------------------#
#K-means
##Load data.

movie <- read.csv("movie_final.csv") 

##Sort the clusters' 10 centers.
centers<-kmeans(movie$new_rev, centers=10, nstart = 20)$centers
centers<-sort(centers)

##Create a new column called "cluster" based on the centers.
km.out<-kmeans(movie$new_rev, centers=centers)
cluster<-km.out$cluster
movie$cluster<-cluster

##Plot the clusters.
plot(movie$new_rev, col = (km.out$cluster + 1), main = "Revenue Clustering Results with K = 10", 
     pch = 20, cex = 2,xaxt="n",yaxt="n",
     xlab=expression(paste("observation")),
     ylab=expression(paste('revenue')))
axis(1,at=c(0, 1000, 2000, 3000, 4000),labels=c('0', '1000', '2000', '3000', '4000'))
axis(2,at=c(0, 200000000,400000000,600000000,800000000),labels=c('0', '200m', '400m', '600m', '800m'))

##Write the data set to csv.
##write.csv(movie,'movie_final.csv')


#########################################################################
#------------------------- Feature engineering -------------------------#
movies_new <- read.csv("movie_final.csv")
movies_new <- movies_new[,c(-1, -2, -3, -4, -9, -35)] # delete irrelevant features such as index, id, imbd id
movies_new <- movies_new[!is.na(movies_new['actor_popularity']),] # delete rows missing actor popularity
movies_new <- movies_new %>% filter(director_popularity<30) # delete director outlier
movies_new$cluster <- as.factor(movies_new$cluster)
for(i in 5:23){
    movies_new[,i] <- as.character(movies_new[,i])
}
movies_new$month <- as.character(movies_new$month)
movies_new$company_class <- factor(movies_new$company_class, levels = c("small","medium","big"))
movies_new$actor_popularity <- as.numeric(movies_new$actor_popularity)
movies_new$director_popularity <- as.numeric(movies_new$director_popularity)


set.seed(123)
movies_new_split <- initial_split(movies_new, prop = 0.7, strata = "cluster")
movies_new_train <- training(movies_new_split)
movies_new_test <- testing(movies_new_split)


movies_new_blueprint <- recipe(cluster~., data = movies_new_train) %>% 
    step_nzv(all_nominal()) %>% 
    step_YeoJohnson(company_count,actor_popularity,director_popularity,new_budget) %>% 
    step_integer(company_class) %>% #order
    step_center(all_numeric(),-all_outcomes()) %>% 
    step_scale(all_numeric(),-all_outcomes()) %>% 
    step_dummy(all_nominal(), -all_outcomes())

movies_new_prepare <- prep(movies_new_blueprint, training = movies_new_train)
movies_new_bake_train <- bake(movies_new_prepare, new_data = movies_new_train)
movies_new_bake_test <- bake(movies_new_prepare, new_data = movies_new_test)



############################################################
#------------------------- MODELS -------------------------#

# -----------------1. logistic regression----------------- #

cv <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 5
)

set.seed(123)
logistic_fit <- train(
    movies_new_blueprint,
    data = movies_new_train,
    method = 'polr',
    trControl = cv,
    metric = "Accuracy"
)

logistic_fit



# -----------------2. RF ----------------- #
movies_new_response <- 'cluster'
movies_new_predictors <- setdiff(colnames(movies_new_bake_train), movies_new_response)
movies_new_n_features <- length(movies_new_predictors)

cv <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 5
)

movies_rf_hyper_grid_train <-  expand.grid(
    mtry = floor(movies_new_n_features * c(.05, .15, .25, .333, .4)),
    min.node.size = c(1,3,5,10),
    splitrule = "gini")

set.seed(123)
movies_RF_fit <- train(
    movies_new_blueprint,
    data = movies_new_train,
    method = 'ranger',
    trControl = cv,
    metric = "Accuracy",
    tuneGrid = movies_rf_hyper_grid_train
)

movies_RF_fit_top5 <- movies_RF_fit$results %>% arrange(desc(Accuracy)) %>% head(5)
movies_RF_fit_top5


# -----------------3. KNN----------------- #
cv <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 5
)
##Set the grid for hyperparameter tuning.
hyper_grid<-expand.grid(k=seq(2,25,by=1))

##Using caret package to tune hyperparameters for the highest accuracy.
set.seed(123)
knn_fit<-train(movies_new_blueprint,
               data=movies_train,
               method='knn',
               trControl=cv,
               tuneGrid=hyper_grid,
               metric="Accuracy"
)
##KNN Results.
knn_fit
###Accuracy was used to select the optimal model using the largest value.
###The final value used for the model was k = 18.


# -----------------4. Naive Bayes----------------- #


bayes_grid <- expand.grid(
    usekernel = TRUE,
    fL = 1, #Laplace
    adjust = seq(0,3,0.5)
)

set.seed(123)
bayes_fit<-train(blueprint,
                 data=movies_train,
                 method='nb',
                 trControl=cv,
                 tuneGrid=bayes_grid,
                 metric="Accuracy")

##Naive Bayes Results.
bayes_fit



# -----------------5. SVM------------------------#
# create resampling method
cv <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 5
)
#####SVM with linear kernel#####

# create a hyperparameter grid search
SVM_Linear_hyper_grid <- expand.grid(
    C = c(0.01, 0.1, 1, 10, 100)
)
# set seed
set.seed(123)
# execute grid search with svmlinear model, use accuracy as preferred metric
SVM_Linear_fit <- train(
    movies_new_blueprint,
    data = movies_train,
    method = 'svmLinear',
    trControl = cv,
    metric = "Accuracy",
    tuneGrid = SVM_Linear_hyper_grid
)
# print model results
SVM_Linear_fit

#####SVM with radial kernel#####

# create a hyperparameter grid search
SVM_Radial_hyper_grid <- expand.grid(
    sigma = c(0.1,1,2),
    C = 10^(-1:1)
)
# set seed
set.seed(123)
# execute grid search with svmradial model, use accuracy as preferred metric
SVM_Radial_fit <- train(
    movies_new_blueprint,
    data = movies_train,
    method = 'svmRadial',
    trControl = cv,
    metric = "Accuracy",
    tuneGrid = SVM_Radial_hyper_grid
)
# print model results
SVM_Radial_fit

#####SVM with poly kernel#####

# create a hyperparameter grid search
SVM_Poly_hyper_grid <- expand.grid(
    degree = c(1,2,3),
    C = c(0.1,1),
    scale = c(0.1,1)
)
# set seed
set.seed(123)
# execute grid search with svmpoly model, use accuracy as preferred metric
SVM_Radial_fit <- train(
    movies_new_blueprint,
    data = movies_train,
    method = 'svmPoly',
    trControl = cv,
    metric = "Accuracy",
    tuneGrid = SVM_Poly_hyper_grid 
)
# print model results
SVM_Radial_fit




# ---------------------- 6. XGBoost---------------------- #


# create a hyperparameter grid search
xgb_hyper_grid <- expand.grid(
    max_depth = c(1,3,5),
    eta =c(0.1,0.3),
    gamma=c(0.1,1),
    min_child_weight=c(1,3),
    nrounds=1000,
    colsample_bytree=0.8,
    subsample=0.5
)
# set seed 
set.seed(123)
# execute grid search with xgboost model, use accuracy as preferred metric
xgb_fit <- train(
    movies_new_blueprint,
    data = movies_train,
    method = 'xgbTree',
    trControl = cv,
    metric = "Accuracy",
    tuneGrid = xgb_hyper_grid
)
# print model results
xgb_fit




# -----------------7. Stochastic GBM----------------- #

# create a hyperparameter grid search
sgb_hyper_grid <- expand.grid(
    n.trees=100,
    interaction.depth=c(3,5),
    shrinkage =c(0.01,0.1),
    n.minobsinnode=5
)
# set seed
set.seed(123)
# execute grid search with gbm model, use accuracy as preferred metric
sgb_fit <- train(
    movies_new_blueprint,
    data = movies_train,
    method = 'gbm',
    trControl = cv,
    metric = "Accuracy",
    tuneGrid = sgb_hyper_grid
)
# print model results
sgb_fit

#############################################################
#------------------------ Ensemble -------------------------#


library(h2o)
h2o.no_progress()
h2o.init(max_mem_size = "5g")
movies_new_train_h2o <- as.h2o(movies_new_bake_train)
movies_new_response <- 'cluster'
movies_new_predictors <- setdiff(colnames(movies_new_bake_train), movies_new_response)
movies_new_n_features <- length(movies_new_predictors)

#1. base learner RF
### use the best tuned hyperpara. in the previous step
set.seed(123)
movies_bs_rf <- h2o.randomForest(
    training_frame = movies_new_train_h2o,
    x = movies_new_predictors,
    y = movies_new_response,
    nfolds = 10, 
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE, 
    seed = 123,
    ntrees = movies_new_n_features*10,
    min_rows = 10,
    max_depth = 30,
    mtries = 13
)


#2. base learner logistic
### use the best tuned hyperpara. in the previous step
movies_bs_logitsic <- h2o.glm(
    training_frame = movies_new_train_h2o,
    x = movies_new_predictors,
    y = movies_new_response,
    nfolds = 10, 
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE, 
    seed = 123,
    family = "multinomial"
)


#3. base learner xgboost
### use the best tuned hyperpara. in the previous step
movies_bs_xgboost <- h2o.xgboost(
    training_frame = movies_new_train_h2o,
    x = movies_new_predictors,
    y = movies_new_response,
    nfolds = 10, 
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE, 
    seed = 123,
    learn_rate = 0.1,
    max_depth = 1,
    min_split_improvement = 0.1,
    min_rows = 1,
    colsample_bytree = 0.8,
    subsample = 0.5
)

#4 base learner MLP
### use the best tuned hyperpara. in the previous step
movies_bs_dl <- h2o.deeplearning(
    training_frame = movies_new_train_h2o,
    x = movies_new_predictors,
    y = movies_new_response,
    nfolds = 10, 
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE, 
    seed = 123,
    adaptive_rate = FALSE,
    activation=c("Rectifier"),
    hidden = c(30),
    rate_decay = 0.9,
    distribution = "multinomial",
    rate = 0.001,
    momentum_start=0.5,             ## manually tuned momentum
    momentum_stable=0.9, 
    momentum_ramp=1e7,
    epochs=50
)

# 5. ensemble
set.seed(12)
movies_ensemble <- h2o.stackedEnsemble(
    training_frame = movies_new_train_h2o,
    x = movies_new_predictors,
    y = movies_new_response,
    base_models = list(movies_bs_rf,movies_bs_logitsic,movies_bs_xgboost,movies_bs_dl),
    metalearner_algorithm = "drf",
    seed = 12
)

movies_new_test_h2o <- as.h2o(movies_new_bake_test)
h2o.performance(movies_ensemble, newdata = movies_new_test_h2o)

#############################################################
#--------------------------- EDA ---------------------------#

p11<-ggplot(movies_new, aes(x=new_budget)) +
    geom_histogram(fill="black", alpha = 0.7)+
    theme_bw()+
    labs(x = "Movie budget", y= "Count")

p12<-ggplot(movies_new, aes(x=actor_popularity)) +
    geom_histogram(fill="black", alpha = 0.7)+
    theme_bw()+
    labs(x = "Actor Popularity", y= "Count")

p13<-ggplot(movies_new, aes(x=actor_popularity)) +
    geom_histogram(fill="black", alpha = 0.7)+
    theme_bw()+
    labs(x = "Actor Popularity", y= "Count")

p14<-ggplot(movies_new, aes(x=director_popularity)) +
    geom_histogram(fill="black", alpha = 0.7)+
    theme_bw()+
    labs(x = "Director Popularity", y= "Count")

grid.arrange(p11, p12, p13, p14, nrow = 2)

##train_bake_data

ggplot(movies_new_bake_train, aes(x=new_budget)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "Movie budget", y= "Count", title = "Distribution of Movie Budget")

ggplot(movies_new_bake_train, aes(x=actor_popularity)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "Actor Popularity", y= "Count", title = "Distribution of Actor Popularity")

ggplot(movies_new_bake_train, aes(x=director_popularity)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "Director Popularity", y= "Count", title = "Distribution of Director Popularity")

ggplot(movies_new_bake_train, aes(x=company_count)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "#Production Company", y= "Count", title = "Distribution of #Production Company")


##test_bake_data

ggplot(movies_new_bake_test, aes(x=new_budget)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "Movie budget", y= "Count", title = "Distribution of Movie Budget")

ggplot(movies_new_bake_test, aes(x=actor_popularity)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "Actor Popularity", y= "Count", title = "Distribution of Actor Popularity")

ggplot(movies_new_bake_test, aes(x=director_popularity)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "Director Popularity", y= "Count", title = "Distribution of Director Popularity")

ggplot(movies_new_bake_test, aes(x=company_count)) +
    geom_histogram(fill="#FFC100", alpha = 0.7)+
    theme_bw()+
    labs(x = "#Production Company", y= "Count", title = "Distribution of #Production Company")

#correlation heatmap

movies_new2<-movies_new[,-c(30)]
library(dplyr)
movies_new %>%
    select_if(is.numeric) %>%
    cor()

library(ggplot2)
ggplot(data =movies_new2%>% select_if(is.numeric) %>% cor() %>% reshape2::melt(),
       aes(x = Var1 ,y = Var2, fill = value)) +
    geom_tile(color="white",size=0.1) +
    xlab("") +
    ylab("") +
    guides(fill = guide_legend(data.crdle = "")) +
    scale_fill_gradient( low = "grey", high = "black")
    theme(axis.text.x = element_text(angle = 25, hjust = 1))

movies_new$cluster<-as.numeric(movies_new$cluster)
#Interaction between categorical explanatory variables and revenue clusters.
p1 <- ggplot(movies_new, aes(x=holiday, y = cluster, fill = holiday))+
    geom_boxplot(alpha = 0.7)+
    theme_bw()+
    labs(x = "If Released During Holiday", y= "Movie Cluster", fill = "holiday")+ 
    theme(axis.text.x=element_text(angle=90, hjust = 1, vjust = 0))+
    theme(legend.position="none")+
    scale_fill_manual(values = c('black','#FFC100'))+
    scale_y_continuous(breaks=seq(1,10,1))

p2 <- ggplot(movies_new, aes(x=sequel, y = cluster, fill = sequel))+
    geom_boxplot(alpha = 0.7)+
    theme_bw()+
    labs(x = "If Is in a Sequel", y= "Movie Cluster", fill = "sequel")+ 
    theme(axis.text.x=element_text(angle=90, hjust = 1, vjust = 0))+
    theme(legend.position="none")+
    scale_fill_manual(values = c('black','#FFC100'))+
    scale_y_continuous(breaks=seq(1,10,1))

p3<-ggplot(movies_new, aes(x=new_language, y = cluster, fill = new_language))+
    geom_boxplot(alpha = 0.7)+
    theme_bw()+
    labs(x = "If Is in English", y= "Movie Cluster", fill = "new_language")+ 
    theme(axis.text.x=element_text(angle=90, hjust = 1, vjust = 0))+
    theme(legend.position="none")+
    scale_fill_manual(values = c('#FFC100','black'))+
    scale_y_continuous(breaks=seq(1,10,1))

p4<-ggplot(movies_new, aes(x=company_class, y = cluster, fill = company_class))+
    geom_boxplot(alpha = 0.7)+
    theme_bw()+
    labs(x = "Production Company Class", y= "Movie Cluster", fill = "company_class")+ 
    theme(axis.text.x=element_text(angle=90, hjust = 1, vjust = 0))+
    theme(legend.position="none")+
    scale_fill_manual(values = c('black',	'#FFED97','#FFC100'))+
    scale_y_continuous(breaks=seq(1,10,1))

grid.arrange(p1, p2, p3, p4, nrow = 2)
