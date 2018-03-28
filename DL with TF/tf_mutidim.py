import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Scalar = tf.constant([2])
Vector = tf.constant([1,2,3,4])
Matrix = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
Tensor = tf.constant([ [[1,2,3], [4,5,6], [7,8,9]], [[10,11,12], [13,14,15], [16,17,18]], [[19,20,21], [22,23,24], [25,26,27]] ])

with tf.Session() as session:
	result = session.run(Scalar)
	print(result)
	result = session.run(Vector)
	print(result)
	result = session.run(Matrix)
	print(result)
	result = session.run(Tensor)
	print(result)


matrix_one = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
matrix_two = tf.constant([[5,2,5], [7,5,8], [6,8,9]])

add_one = tf.add(matrix_one, matrix_two)
add_two = matrix_one + matrix_two
mul_operation = tf.matmul(matrix_one, matrix_two)

with tf.Session() as session:
	result = session.run(add_one)
	print(result)
	result = session.run(add_two)
	print(result)
	result = session.run(mul_operation)
	print(result)

state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_variable = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init_variable)
	for _ in range(3):
		session.run(update)
		print(session.run(state))


