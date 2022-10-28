import tensorflow as tf
tf.compat.v1.set_random_seed(1234)


variable = tf.compat.v1.Variable(tf.random_normal([1]))
print(variable)


#1. 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(variable)
print('aaa :', aaa) # aaa : [-0.13862522]
sess.close() 



#2. 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = variable.eval(session=sess) # 이 과정을 거처야 진정한 변수로 다시 태어난다
print('bbb :', bbb) # bbb : [-0.13862522]
sess.close() 



#3. 초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = variable.eval()
print('ccc :', ccc) # ccc : [-0.13862522]
sess.close()