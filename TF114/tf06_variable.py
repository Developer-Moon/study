import tensorflow as tf


sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer() # 변수를 초기화 시킨다 (외 몇가지가 더 있다)
# 그 전에 했던 모든 변수들에 대해서 초기화를 해준다 [초기값을 넣을 수 있는 상태가 된다. 이 코드 위로만 지정해준다 하단은 X]
sess.run(init) # 꼭 실행시켜줘야 한다

print(sess.run(x + y)) # 5.0