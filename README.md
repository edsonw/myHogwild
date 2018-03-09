# 实现论文 “HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent”
根据HOGWILD!的思想，通过基于sklearn 的joblib 来实现多线程的随机梯度下降
Generators：
Return the data set to each threads

Shared：
Create the shared weights, and use each batch of data to update the shared weights.

Hogwildsgd：
This class implements hogwild!. The main function including fit and predict. The fit function uses the multiple threads to update the shared weights.This class we used the sklearn- joblib for creating the threads.

Test：
This class is the example of how we use hogwild! and how we prepare the data set.

Environment:
Python3.6 +numpy+sklearn

Method of Run：
In the folder src ,open terminal then input command :
python test.py
