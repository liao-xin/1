
import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk
import numpy as np
import jieba
import data_input_helper as data_helpers

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

#综合图像识别和文字分类
import numpy as np
import os, sys
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *
import numpy as np
import os, sys
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *
import warnings
warnings.filterwarnings("ignore")

window = tkinter.Tk()
window.title('垃圾分类识别界面')
window.geometry('800x1000')

#下面这个可以展开全屏
window.state("zoomed")

#固定窗口，使界面不可放大或缩小
#window.resizable(0, 0)
var1 = tkinter.StringVar()
#弄个花里花俏一点的界面增加视觉效果而已
T = tkinter.Label(window, text="垃圾分类",font=("微软雅黑", 20), textvariable=var1, bg="lightGreen", fg="DimGray", anchor="se")

T.place(x=0, y=0, width=1500, height=120)

#显示图片路径以及识别结果的窗口
tkinter.Label(window, text='请输入识别文本: ', font=("微软雅黑", 25)).place(x=50, y=200)
tkinter.Label(window, text='文本识别结果为: ', font=("微软雅黑", 25)).place(x=50, y=300)

tkinter.Label(window, text='请输入识别图片: ', font=("微软雅黑", 25)).place(x=50, y=400)
tkinter.Label(window, text='图片识别结果为: ', font=("微软雅黑", 25)).place(x=50, y=500)

var_user_name = tkinter.StringVar()
entry_user_name = tkinter.Entry(window, textvariable=var_user_name, font=("微软雅黑", 15))
entry_user_name.place(x=300, y=220, width=300, height=30)
var_user_pd = tkinter.StringVar()
entry_user_pd = tkinter.Entry(window, textvariable=var_user_pd, font=("微软雅黑", 15))
entry_user_pd.place(x=300, y=320, width=300, height=30)

var_load = tkinter.StringVar()
entry_load = tkinter.Entry(window, textvariable=var_load, font=("微软雅黑", 15))
entry_load.place(x=300, y=420, width=300, height=30)
var_s = tkinter.StringVar()
entry_s = tkinter.Entry(window, textvariable=var_s, font=("微软雅黑", 15))
entry_s.place(x=300, y=520, width=300, height=100)
'''
var_load2 = tkinter.StringVar()
entry_load2 = tkinter.Entry(window, textvariable=var_load2, font=("微软雅黑", 20))
entry_load2.place(x=1060, y=370, width=400, height=50)
var_s2 = tkinter.StringVar()
entry_s2 = tkinter.Entry(window, textvariable=var_s2, font=("微软雅黑", 20))
entry_s2.place(x=1060, y=470, width=400, height=50)
'''

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


class RefuseClassification2():

    def __init__(self):
        self.w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)  # 加载词向量
        self.init_model()
        self.refuse_classification_map = {0: '可回收垃圾', 1: '有害垃圾', 2: '湿垃圾', 3: '干垃圾'}

    def deal_data(self, text, max_document_length=10):
        words = jieba.cut(text)
        x_text = [' '.join(words)]
        x = data_helpers.get_text_idx(x_text, self.w2v_wr.model.vocab_hash, max_document_length)

        return x

    def init_model(self):
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            self.sess.as_default()
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)

            # Get the placeholders from the graph by name
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]

            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def predict(self, text):
        x_test = self.deal_data(text, 5)
        predictions = self.sess.run(self.predictions, {self.input_x: x_test, self.dropout_keep_prob: 1.0})

        refuse_text = self.refuse_classification_map[predictions[0]]
        return refuse_text





class RafuseRecognize():

    def __init__(self):
        self.refuse_classification = RefuseClassification()
        self.init_classify_image_model()
        self.node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt',
                                      model_dir='/tmp/imagenet')

    def init_classify_image_model(self):
        create_graph('/tmp/imagenet')

        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')

    def recognize_image(self, image_data):
        predictions = self.sess.run(self.softmax_tensor,
                                    {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        result_list = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)
            # print(human_string)
            human_string = ''.join(list(set(human_string.replace('，', ',').split(','))))
            # print(human_string)
            classification = self.refuse_classification.predict(human_string)
            result_list.append('%s  =>  %s' % (human_string, classification))

        return '\n'.join(result_list)

# 打开文件函数
def choose_fiel():
    selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
    var_load.set(selectFileName)


# 识别图片数字函数
def main(img):
    test = RafuseRecognize()
    image_data = tf.gfile.FastGFile(img, 'rb').read()
    res = test.recognize_image(image_data)
    var_s.set(res)

def main1():
    test = RefuseClassification()
    res = test.predict(entry_user_pd.get())
    var_user_pd.set(res)

# 清零函数，不过没啥意义，就想把界面弄的对称一点而已
def delete():  # 删除函数
    content = var_user_pd.get()
    var_user_pd.set(content[0:len(content) - 1])


# 显示所要识别的图片函数
def showImg(img1):
    # canvas = tkinter.Canvas(window, height=400, width=1000)
    load = Image.open(img1)
    render = ImageTk.PhotoImage(load)
    img = tkinter.Label(image=render)
    img.image = render
    # canvas.create_image(0, 0, anchor='nw', image=render)
    # canvas.pack(side='bottom')
    img.place(x=160, y=120)


# 按钮
'''
submit_button = tkinter.Button(window, text="选择文件", command=choose_fiel).place(x=50, y=250)
submit_button = tkinter.Button(window, text="文字识别", command=delete()).place(x=50, y=300)
submit_button = tkinter.Button(window, text="显示图片", command=lambda: showImg(entry_user_name.get())).place(x=219, y=250)
submit_button = tkinter.Button(window, text="图片识别", command=lambda: main(entry_user_name.get())).place(x=219, y=300)
'''
submit_button = tkinter.Button(window, text="文本识别",font=("微软雅黑", 20), command=lambda: main1()).place(x=800, y=200)

submit_button = tkinter.Button(window, text="选择图片", font=("微软雅黑", 20), command=choose_fiel).place(x=800, y=300)
submit_button = tkinter.Button(window, text="图片识别", font=("微软雅黑", 20),command=lambda: main(var_load.get())).place(x=800, y=400)

submit_button = tkinter.Button(window, text="结束程序",font=("微软雅黑", 20), command=lambda: main1()).place(x=800, y=500)
window.mainloop()