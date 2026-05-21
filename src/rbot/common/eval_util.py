

class SimpleRecorder:
    def __init__(self, dir) -> None:
        self.pool = multiprocessing.Pool(8)
        dir = os.path.join(dir, f'{int(time.time())}')
        self.dir = dir
        os.makedirs(dir)
        with open(os.path.join(dir, 'info.txt'), 'w') as f:
            f.write(f'{sys.argv}')

    def save(self, image):
        path = os.path.join(self.dir, f'{int(time.time()*1000)}.png')
        self.pool.apply_async(self.save_image, args=[image, path])

    @staticmethod
    def save_image(image, path):
        Image.fromarray(image).save(path)
