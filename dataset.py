from data_reader import DataReader


# 'rooms_ring_camera'
def get_dataset(path, name, context_size, batch_size, training):
    mode = 'train' if training else 'test'

    train_data_reader = DataReader(name, context_size, path, mode)
    data_train = train_data_reader.read(batch_size)

    x = data_train.query.context.frames
    v = data_train.query.context.cameras
    x_q = data_train.target
    v_q = data_train.query.query_camera

    return (x, v), (x_q, v_q)
