# coding:utf-8

import requests
from bs4 import BeautifulSoup
import tensorflow as tf
import concurrent.futures
import time
from util import path

spiders_dic = {
    'sport': "https://search.jd.com/Search?keyword=%E8%BF%90%E5%8A%A8%E8%A3%85%E5%A5%B3&enc=utf-8&page="
}
search_page_num = 4

class spider:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
        self.pids = set()  # 页面中所有的id,用来拼接剩下的30张图片的url,使用集合可以有效的去重
        self.img_urls = set()  # 得到的所有图片的url

    # 得到每一页的网页源码
    def get_html(self, page, style):
        url = spiders_dic[style] + str(page)
        res = requests.get(url, headers=self.headers)
        html = res.text
        return html

    # 得到每一个页面的id
    def get_pids(self, page, style):
        html = self.get_html(page, style)
        soup = BeautifulSoup(html, 'lxml')
        lis = soup.find_all("li", class_='gl-item')
        for li in lis:
            data_pid = li.get("data-pid")
            if (data_pid):
                self.pids.add(data_pid)
                # print self.pids
                # print "-------------------------------------------------------------"

    def get_src_imgs_data(self, page, style):
        html = self.get_html(page, style)
        soup = BeautifulSoup(html, 'lxml')
        divs = soup.find_all("div", class_='p-img')  # 图片
        # divs_prices = soup.find_all("div", class_='p-price')   #价格
        image_src = []
        for div in divs:
            img_1 = div.find("img").get("src")  # 得到已经加载出来的url
            img_2 = div.find("img").get("source-data-lazy-img")
            if img_1:
                # image_src.append('http:' + img_1)
                pass
            if img_2:
                image_src.append('http:' + img_2)
        print("============Get size of image : %d" % len(image_src))
        return image_src

    def do_download(self, image_src, cache_dir, cache_subdir):
        for image_urls in image_src:
            for image_url in image_urls:
                while True:
                    download_ok = True
                    try:
                        image_name = image_url.split('/')[-1]
                        tf.keras.utils.get_file(fname=image_name, origin=image_url,
                                                cache_dir=cache_dir,
                                                cache_subdir=cache_subdir)
                    except Exception as e:
                        download_ok = False
                        print(e)
                        print("start retry")
                    if download_ok:
                        break
    def save_image2file(self, style, photo_save_dir, photo_save_subdir, thread_number):
        images_src = []
        for i in range(thread_number):
            images_src.append([])
        for page in range(search_page_num):
            images_src[page % thread_number].append(self.get_src_imgs_data(page, style))
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_number) as executor:
            futures = []
            start_time = time.time()
            for image_src in images_src:
                futures.append(executor.submit(self.do_download(image_src, photo_save_dir, photo_save_subdir)))
            concurrent.futures.wait(futures)
            print("Time of downloading pictures : %f s" % (time.time() - start_time))

    def main(self, style, photo_save_dir, photo_save_subdir, thread_number):
        self.save_image2file(style, photo_save_dir, photo_save_subdir, thread_number)
        print ("--------------------------------------SUCCESS----------------------------------------------")

if __name__ == '__main__':
    style = 'sport'
    photo_save_dir = path.SPIDER_PATH
    photo_save_subdir = style
    thread_number = 4
    spider().main(style, photo_save_dir, photo_save_subdir, thread_number)
