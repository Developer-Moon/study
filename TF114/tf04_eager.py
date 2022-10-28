import tensorflow as tf


print(tf.__version__)
print(tf.executing_eagerly())          # False 

#즉시 실행모드!!! (tensor2 를 쓰는데 1문법을 실행시키고 싶을 때)
tf.compat.v1.disable_eager_execution() # 즉시 실행모드를 끄겠다

print(tf.executing_eagerly())          # False - 실행시키지 않았다는 1. 대 버전    true- 실행시켰다는건  2. 대 버전

hello = tf.compat.v1.constant('hello') 

sess = tf.Session()
print(sess.run(hello)) # b'hello