
setwd('C:/Users/Vinay/Documents/STAT5525')
data <- readRDS("Scheetz2006.rds")
attach(data)
n = nrow(X)# Split data into train and test sets
train_rows <- sample(1:n, n * 0.7)
X.train <- X[train_rows, ]
X.test <- X[-train_rows, ]
y.train <- y[train_rows]
y.test <- y[-train_rows]

dim(X.train)
dim(X.test)

#Lasso Regression

#install.packages("glmnet")
library(glmnet)

lasso.mod = glmnet(X.train,y.train,family="gaussian",nlambda=100,alpha=1)
plot(lasso.mod) #L1 Normalization of Lasso Coefficients


set.seed(1)
cv.out = cv.glmnet(X.train, y.train, alpha =1) #Cross validation to find the best lambda
plot(cv.out)

bestlam =cv.out$lambda.min


lasso.pred = predict(lasso.mod, s = bestlam, newx=X.test)
mse_lasso=mean((lasso.pred-y.test)^2) # Get MSE value

lasso.coef=predict(lasso.mod, type="coefficients", s=bestlam)
lasso_num_coef=length(lasso.coef[lasso.coef != 0])

#Horseshoe Regression

#install.packages("horseshoe")
library(horseshoe)

horseshoe.fit <- horseshoe(y=y.train, X=X.train, method.tau = "truncatedCauchy", method.sigma="Jeffreys")

plot(y, X%*%horseshoe.fit$BetaHat) # Predictive values against the Scheetz 2006 data


horseshoe.fit$TauHat #posterior mean of tau

horseshoe.betas <- HS.var.select(horseshoe.fit, y.train, method="interval")
horseshoe_num_coef <- sum(horseshoe.betas) # Number of coefficients chosen
horseshoe_num_coef
horseshoe.fit$BetaHat[horseshoe.betas] #Coefficient values


library(Hmisc)
xyplot(Cbind(horseshoe.fit$BetaHat, horseshoe.fit$LeftCI, horseshoe.fit$RightCI) ~ 1:30) #Credible Intervals

horseshoe.pred <- X.test %*% horseshoe.fit$BetaHat
mse_horseshoe=mean((y.test - horseshoe.pred)^2)

#Adaptive ElasticNet

#install.packages("gcdnet")
library(gcdnet)
adapt_en_model = gcdnet(X.train, y.train, method ="ls")
plot(adapt_en_model) #L1 Normalization of Adaptive ElasticNet Coefficients

set.seed(1)
#Cross validation to find the best lambda
cv.out = cv.gcdnet(X.train, y.train, method ="ls",pred.loss="loss")
plot(cv.out) # Lambda Plot for Adaptive ElasticNet model

#Get the optimum lambda value
bestlam_adapt_en =cv.out$lambda.min
print(bestlam_adapt_en)

#Get the Mean Squared Error
adapt_en_pred=predict(adapt_en_model, s = bestlam_adapt_en , newx = X.test)
mse_adapt_en=mean((adapt_en_pred - y.test)^2)
mse_adapt_en

#Find number of non-zero coefficients
adapt_en.coef=coef(cv.out , s =bestlam_adapt_en , type="coefficients")
adapt_en_num_coef=nnzero(adapt_en.coef)
adapt_en_num_coef

tab <- matrix(c( round(mse_lasso,8), lasso_num_coef, round(mse_horseshoe,8), horseshoe_num_coef, round(mse_adapt_en,8),adapt_en_num_coef), ncol=2, byrow=TRUE)
colnames(tab) <- c('Mean_Squared_Error','Variables_Selected')
rownames(tab) <- c('Lasso','Horseshoe','Adaptive ElasticNet')
tab <- as.table(tab)
tab