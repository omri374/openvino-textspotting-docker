class ResultsLogger:

    def __init__(self):
        self.basedir = './log/'
        self.index = 0

    def log_ocr(self, image, segments, server_ocr, image_id=None):
        self.index += 1
        folder_name = self.basedir + time.strftime("%Y_%m_%d_%H")
        os.makedirs(folder_name, exist_ok=True)
        file_name = f'{folder_name}/{self.index:09}'
        if image_id:
            file_name = f'{folder_name}/{image_id}'

        json.dump({'segments': segments, 'server_ocr': server_ocr}, open(f'{file_name}.json', 'w'))

        if image is not None:
            with open(f'{folder_name}/{file_name}.jpg', 'wb') as f:
                f.write(image)
