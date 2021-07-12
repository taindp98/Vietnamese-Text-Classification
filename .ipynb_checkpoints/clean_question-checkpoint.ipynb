{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import pandas as pd\n",
    "import re\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "uniChars = \"àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ\"\n",
    "unsignChars = \"aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU\"\n",
    "\n",
    "\n",
    "def loaddicchar():\n",
    "    dic = {}\n",
    "    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(\n",
    "        '|')\n",
    "    charutf8 = \"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ\".split(\n",
    "        '|')\n",
    "    for i in range(len(char1252)):\n",
    "        dic[char1252[i]] = charutf8[i]\n",
    "    return dic\n",
    "dicchar = loaddicchar()\n",
    "def convert_unicode(txt):\n",
    "    return re.sub(\n",
    "        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',\n",
    "        lambda x: dicchar[x.group()], txt)\n",
    "def clean_mess(mess):\n",
    "    # input: câu nhập vào của người dùng\n",
    "    # return: câu đã loại bỏ special token\n",
    "    mess_unic = convert_unicode(mess).lower()\n",
    "    mess_rmspectoken = re.findall(r'(?i)\\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ0-9]+\\b', mess_unic)\n",
    "    mess_norm = ' '.join(mess_rmspectoken)\n",
    "    return mess_norm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = '/home/taindp/Jupyter/intent_bert/data/question_livestream_label.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "question = pd.read_csv(data_path)\n",
    "# df = pd.DataFrame([])\n",
    "# df['label'] = question['label'] = [0 for _ in range(len(question))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>content</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>thầy cho em hỏi nếu mình đã trúng tuyển chương...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>cho em hỏi em có thể đăng kí 2 ngành được khôn...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>cho em hỏi chương trình chất lượng cao ở bách ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>cho em hỏi nếu em đã trúng tuyển chương trình ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>thầy ơi cho em hỏi ví dụ nếu mình chọn nguyện ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>428</th>\n",
       "      <td>3</td>\n",
       "      <td>cho em hỏi về ngành kỹ thuật hoá học và cơ hội...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>429</th>\n",
       "      <td>3</td>\n",
       "      <td>cho em xin giới thiệu về ngành kỹ thuật robot ạ</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>430</th>\n",
       "      <td>3</td>\n",
       "      <td>ngành khoa học máy tính sau này ra làm công vi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>431</th>\n",
       "      <td>3</td>\n",
       "      <td>em muốn học tự động hoá thì tương lai sẽ có ng...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>432</th>\n",
       "      <td>3</td>\n",
       "      <td>dạ cho em hỏi về đầu ra và cơ hội nghề nghiệp ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>433 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     label                                            content\n",
       "0        1  thầy cho em hỏi nếu mình đã trúng tuyển chương...\n",
       "1        0  cho em hỏi em có thể đăng kí 2 ngành được khôn...\n",
       "2        1  cho em hỏi chương trình chất lượng cao ở bách ...\n",
       "3        1  cho em hỏi nếu em đã trúng tuyển chương trình ...\n",
       "4        0  thầy ơi cho em hỏi ví dụ nếu mình chọn nguyện ...\n",
       "..     ...                                                ...\n",
       "428      3  cho em hỏi về ngành kỹ thuật hoá học và cơ hội...\n",
       "429      3    cho em xin giới thiệu về ngành kỹ thuật robot ạ\n",
       "430      3  ngành khoa học máy tính sau này ra làm công vi...\n",
       "431      3  em muốn học tự động hoá thì tương lai sẽ có ng...\n",
       "432      3  dạ cho em hỏi về đầu ra và cơ hội nghề nghiệp ...\n",
       "\n",
       "[433 rows x 2 columns]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "question"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "plt.hist(list(question['label']), bins = 10)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "nlp",
   "language": "python",
   "name": "nlp"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
