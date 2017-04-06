# coding=UTF-8
import cPickle
import codecs
import sys

sys.path.append("../")
import numpy as np
from scipy.sparse import csr_matrix

from config.config import FilePathConfig, ClassifierConfig
from evaluation.test_result import TestResult
from feature_extractor.entity.document import Document
from feature_extractor.entity.document_vector import DocumentVector
from feature_extractor.entity.lexicon import Lexicon
from feature_extractor.entity.termweight import TfOnlyTermWeighter, TfIdfWighter
from feature_extractor.feature_filter.speech_filter import SpeechFilter
from feature_extractor.feature_filter.stop_word_filter import StopWordFilter
from feature_extractor.feature_filter.common_filter import CommonFilter
from feature_extractor.feature_selection_functions.chi_square import ChiSquare
from feature_extractor.feature_selection_functions.informantion_gain import InformationGain
from misc.util import Util
from model.abstract_classifier import AbstractClassifier

reload(sys)
sys.setdefaultencoding('utf-8')


class MainClassifier(object):
    def __init__(self):
        # 主要都是直接从配置文件里读取
        # 加载类别与id的映射字典
        self.category_dic = self.load_category_dic_from_pkl()
        # 加载词典
        self.lexicon = self.load_lexicon()
        # 加载特征降维方法
        self.select_function = self.get_selection_funtion()
        # 将训练转换成文档向量的工具
        self.training_vector_builder = DocumentVector(TfOnlyTermWeighter(self.lexicon))
        # 将测试集转换成文档向量的工具
        self.test_vector_builder = DocumentVector(TfIdfWighter(self.lexicon))
        # 类别数量
        self.num_categories = len(self.category_dic)
        # 缓存文件
        self.cache_file = None
        # 这次要使用的分类器
        self.abstract_classifier = AbstractClassifier()
        # 文章最长的长度
        self.longest_length_doc = 0
        # 文章的总数量
        self.num_doc = 0
        # 特征过滤器
        self.filters = []


    # -----------------------------------------------------------------------------------------------------------------
    # 训练前的准备，构造词典，特征降维，准备训练数据
    def construct_lexicon(self, train_corpus_path):
        if Util.is_file(FilePathConfig.lexicon_pkl_path):
            Util.log_tool.log.debug("has lexicon")
            return
        # 从原始语料中转换成稀疏矩阵，保存在cache中，同时会在lexicon里保存下所有的id_dic和name_dic
        self.add_documents_from_file(train_corpus_path)
        # 进行特征降维,返回一个保存了所有特征id的优先队列
        selected_features_queue = self.select_function.feature_select(self.lexicon, self.num_categories, self.num_doc)

        # 获取选择后的特征对应的id，并保存下所有的特征，这里之所以两个功能放在一起做，是由于优先队列出队的不可逆性
        fid_dic = self.output_selected_features_and_get_fid_dic(selected_features_queue)

        # 根据选择后的特征，重新调整词典id的映射关系
        self.lexicon = self.lexicon.map(fid_dic)
        # 当训练好后，则锁定词典,避免再次修改
        self.lexicon.locked = True
        # 保存下词典映射文件
        self.save_lexicon_into_pkl()

    def output_selected_features_and_get_fid_dic(self, selected_features_queue):
        selected_features_file = codecs.open(FilePathConfig.selected_features_path, 'wb', FilePathConfig.file_encodeing,
                                             'ignore')
        fid_dic = dict()
        feature_to_sort = list()
        while selected_features_queue.qsize() > 0:
            term = selected_features_queue.get()
            feature_to_sort.append(term)
            word = self.lexicon.get_word(term.term_id)
            content = word.name + " " + str(word.df) + " " + str(term.weight)
            selected_features_file.write(content + "\n")

        selected_features_file.close()
        sorted(feature_to_sort)
        for i in range(0, len(feature_to_sort)):
            fid_dic[feature_to_sort[i].term_id] = i
        return fid_dic

    # 添加单篇文档用于构造词典
    def add_document(self, raw_document):
        # 将原始数据转换成整齐格式的文档
        document = Document(raw_document)

        # 检查类别是否合法
        if document.label not in self.category_dic:
            Util.log_tool.log.error("Error category error")

        # 如果cache文件还未打开，则打开
        if self.cache_file is None:
            Util.log_tool.log.debug("open file")
            self.cache_file = codecs.open(FilePathConfig.cache_file_path, 'wb', FilePathConfig.file_encodeing, 'ignore')

        # 如果需要对文章的内容进行过滤，则添加词的过滤器
        # if not ClassifierConfig.is_use_bigram:
        #     for feature_filter in self.filters:
        #         document.add_filter(feature_filter)

        # 从文档中拿出我们需要的特征
        content_words = document.get_content_words_feature()
        self.lexicon.add_document(content_words)
        words = self.lexicon.convert_document(content_words)
        terms = self.training_vector_builder.build(words, False, 0)
        try:
            if len(terms) > self.longest_length_doc:
                self.longest_length_doc = len(terms)

            line_result = str(self.category_dic[document.label]) + FilePathConfig.tab
            for term in terms:
                line_result += (str(term.term_id) + FilePathConfig.colon + str(term.weight))
                line_result += FilePathConfig.space
            self.cache_file.write(line_result.strip() + '\n')
        except:
            Util.log_tool.log.error("Error write cache error when add document")

        self.num_doc += 1

    # 添加多篇文档，循环调用添加单篇文档
    def add_documents(self, raw_documents):
        count = 0
        for raw_document in raw_documents:
            if count % 10000 == 0:
                Util.log_tool.log.debug("加载" + str(count))
            count += 1
            self.add_document(raw_document)

        self.close_cache()

    # 从文件添加多篇文档，循环调用添加单篇文档
    def add_documents_from_file(self, raw_documents_file_path):
        raw_documents = codecs.open(raw_documents_file_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        self.add_documents(raw_documents)
        raw_documents.close()

    # -----------------------------------------------------------------------------------------------------------------
    # 分类相关
    # 分类单篇文档，返回多个结果
    def classify_document_top_k(self, feature_mat, k):
        return self.abstract_classifier.classify_top_k(feature_mat, k)

    # 分类多篇文档，循环调用分类单篇文档，返回多个结果
    def classify_documents_top_k(self, feature_mat, k):
        return self.abstract_classifier.classify_top_k(feature_mat, k)

    # 从文件分类多篇文档，循环调用分类单篇文档，返回多个结果
    def classify_documents_top_k_from_file(self, raw_documents_file_path, k):
        if Util.is_file(FilePathConfig.raw_feature_path):
            print "load raw mat"
            feature_mat, label_vec = Util.get_libsvm_data(FilePathConfig.raw_feature_path)
        else:
            feature_mat = self.corpus_to_feature_mat_from_file(raw_documents_file_path)
        classify_results = self.classify_documents_top_k(feature_mat, k)
        return classify_results

    # 分类单篇文档,只返回一个结果
    def classify_document(self, feature_mat):
        return self.classify_document_top_k(feature_mat, 1)

    # 分类多篇文档，循环调用分类单篇文档,只返回一个结果
    def classify_documents(self, feature_mat):
        return self.classify_documents_top_k(feature_mat, 1)

    # 从文件分类多篇文档，循环调用分类单篇文档,只返回一个结果
    def classify_documents_from_file(self, raw_documents_file_path):
        return self.classify_documents_top_k_from_file(raw_documents_file_path, 1)

    # -----------------------------------------------------------------------------------------------------------------
    # 训练和评测相关
    # 打印分类结果与评测结果
    def print_classify_result(self, predicted_class, raw_class_label):
        labels = sorted(self.category_dic.iteritems(), key=lambda key_value: key_value[1])
        labels = [labels_id_pair[0] for labels_id_pair in labels]
        test_result = TestResult(predicted_class, raw_class_label, labels)
        test_result.print_report()

    def train(self, train_corpus_path):
        Util.log_tool.log.debug("train")
        self.set_model()
        train_feature_mat, label_vec = self.corpus_to_feature_and_label_mat(train_corpus_path,
                                                                            FilePathConfig.train_feature_mat_path)
        self.abstract_classifier.train(train_feature_mat, label_vec)

    def test(self, test_corpus_path):
        Util.log_tool.log.debug("test")
        self.set_model()
        test_sparse_mat, label_vec = self.corpus_to_feature_and_label_mat(test_corpus_path,
                                                                          FilePathConfig.test_feature_mat_path)
        predicted_class_and_pro = self.classify_documents(test_sparse_mat)
        predicted_class = [class_and_pro[0][0] for class_and_pro in predicted_class_and_pro]
        self.print_classify_result(predicted_class, label_vec)

    def corpus_to_feature_mat_from_file(self, corpus_path):
        data = codecs.open(corpus_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        sparse_mat = self.data_to_feature(data)
        Util.save_svmlight_file(sparse_mat, np.zeros(sparse_mat.shape[0]), FilePathConfig.raw_feature_path)
        data.close()
        return sparse_mat

    def data_to_feature(self, data):
        row = list()
        col = list()
        weight = list()
        row_num = 0
        for line in data:
            print row_num
            document = Document(line)
            content_words = document.get_content_words_feature()
            doc_len = len(content_words)
            words = self.lexicon.convert_document(content_words)
            terms = self.test_vector_builder.build(words, True, doc_len)
            terms.sort(cmp=lambda x, y: cmp(x.term_id, y.term_id))
            for term in terms:
                row.append(row_num)
                col.append(term.term_id)
                weight.append(term.weight)
            row_num += 1
        sparse_mat = csr_matrix((np.array(weight), (np.array(row), np.array(col))),
                                shape=(row_num, ClassifierConfig.max_num_features))
        return sparse_mat

    # 将传进来的批量json转换为可用于分类的特征向量矩阵,或者特征向量加原来的分类标签
    def corpus_to_feature_and_label_mat(self, corpus_path, result_path):
        if Util.is_file(result_path):
            Util.log_tool.log.debug("loading data")
            return Util.get_libsvm_data(result_path)
        data = codecs.open(corpus_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        sparse_mat = codecs.open(result_path, 'wb', FilePathConfig.file_encodeing, 'ignore')
        count = 0
        for line in data:
            count += 1
            if count % 10000 == 0:
                Util.log_tool.log.debug("add" + str(count))
            document = Document(line)
            label_id = self.category_dic[document.label]
            content_words = document.get_content_words_feature()
            doc_len = len(content_words)

            words = self.lexicon.convert_document(content_words)
            terms = self.test_vector_builder.build(words, True, doc_len)

            sparse_mat.write(str(label_id))
            # 将id_weight对按照id大小，从小到大排列
            terms.sort(cmp=lambda x, y: cmp(x.term_id, y.term_id))
            for term in terms:
                sparse_mat.write(" " + str(term.term_id) + ":" + str(term.weight))
            sparse_mat.write("\n")

        data.close()
        sparse_mat.close()
        return Util.get_libsvm_data(result_path)

    def set_model(self):
        if ClassifierConfig.is_single_model:
            self.abstract_classifier.model_path = ClassifierConfig.classifier_path_dic[
                ClassifierConfig.cur_single_model]
        else:
            Util.log_tool.log.debug("not single model")

    def load_lexicon(self):
        if Util.is_file(FilePathConfig.lexicon_pkl_path):
            Util.log_tool.log.debug("load lexicon")
            return Util.load_object_from_pkl(FilePathConfig.lexicon_pkl_path)
        else:
            return Lexicon()

    # 从文件加载字典对象
    def load_lexicon_from_pkl(self):
        return Util.load_object_from_pkl(FilePathConfig.lexicon_pkl_path)

    # 加载类别与编号字典,字典内容类似为{"时政":1,"军事":2,……}
    def load_category_dic_from_pkl(self):
        return Util.load_object_from_pkl(FilePathConfig.category_pkl_path)

    def save_lexicon_into_pkl(self):
        cPickle.dump(self.lexicon, open(FilePathConfig.lexicon_pkl_path, 'wb'))

    # 调整boosting的权重　
    def adapt_boosting_weight(self):
        pass

    # 选择是哪种特征降维方式
    def get_selection_funtion(self):
        if ClassifierConfig.cur_selection_function == ClassifierConfig.chi_square:
            return ChiSquare()
        elif ClassifierConfig.cur_selection_function == ClassifierConfig.information_gain:
            return InformationGain()
        else:
            Util.log_tool.log.error("get_selection_funtion error")

    def close_cache(self):
        # 在需要的时候关闭cache文件
        if self.cache_file is not None:
            Util.log_tool.log.debug("close cache")
            self.cache_file.close()

    def init_filter(self):
        common_filter = CommonFilter()
        stop_words_filter = StopWordFilter()
        speech_filter = SpeechFilter()
        self.filters.append(common_filter)
        self.filters.append(stop_words_filter)
        self.filters.append(speech_filter)


def main1():
    # 训练和评测阶段，这里把所有可能需要自定义的参数全部都移到配置文件里了，如果需要也可以换成传参调用的形式
    # 需要外面传进来的参数只有训练集的位置和验证集的位置
    mainClassifier = MainClassifier()
    Util.log_tool.log.debug("lexicon locked:" + str(mainClassifier.lexicon.locked))
    # # 根据原始语料进行语料预处理（切词、过滤、特征降维）
    mainClassifier.construct_lexicon(FilePathConfig.total_corpus_path)
    # # 训练
    mainClassifier.train(FilePathConfig.train_corpus_path)
    # # 测试
    mainClassifier.test(FilePathConfig.test_corpus_path)


def main2():
    # ----------------------------------------------------------------------------------------------------
    # 对外来的数据进行分类
    mainClassifier = MainClassifier()
    mainClassifier.set_model()
    # 需要分类的数据
    # 数据的编码问题需要认真研究
    raw_document = [
        u'{"ID":"110","url":"http://news.ifeng.com/a/20150330/43444856_0.shtml","title":"柯文哲纠正司仪“恭请市长”说法：别用封建语言","splitTitle":"柯文哲_nr 纠正_v 司仪_n 恭请_v 市长_x 说法_k ：_w 别_k 用_k 封建_k 语言_n ","splitContent":"  _w 原_k 标题_n ：_w 纠正_v 恭请_v  _w 柯文哲_nr ：_w 别_k 用_k 封建_k 语言_n  _w \n_w  _w \n_w  _w 台海网_n 3月_t 30日_t 讯_k  _w 据_h 中国时报_x 报道_k ，_w 台北市府_x 29日_mq 举办_v 首长_n 领航_k 营_k ，_w 司仪_n 刚_d 宣布_v 恭请_v 市长_x 致词_k 后_f ，_w 旋_k 遭_k 柯文哲_nr 打_k 脸_n 不要_v 再_d 用_k 这种_mq 封建_k 时代_n 的_u 语言_n ，_w 引起_v 现场_n 哄堂大笑_i 。_w 但_k 卫生局_x 在_p 讲义_n 中_n 整理_v 40条_mq 柯_k 语录_x ，_w 连_h 奇怪_a 耶_k ，_w 听_k 起来_v 怪怪的_z 都_n 入列_v ，_w 卫生局长_x 黄世杰_nr 澄清_k ，_w 市长_x 的_u 核心_n 价值_n 很_d 重要_a ，_w 才_h 会_v 整理_v 市长_x 说_v 过_v 的_u 重点_h ，_w 纯属_v 参考_v 不用_v 牢记_v 或_d 盲从_v 。_w  _w \n_w  _w \n_w  _w 柯文哲_nr 透露_v ，_w 有_v 次_q 去_n 参加_v 国民党_x 一个_mq 会议_n ，_w 他们_r 很_d 喜欢_v 互_d 称_v 什么_r 公_k 、_w 什么_r 公_k ，_w 像_k 吴伯雄_x 就_d 被_p 称为_v 伯公_n 。_w 因此_c 有_v 次_q 遇到_v 民进党_x 主席_x 蔡英文_nr ，_w 自己_h 就_d 和_c 她_r 提起_v 这件_mq 事_k ，_w 并_h 说_v 还好_v 民进党_x 没有_v 这种_mq 文化_x ，_w 不然_c 像_k 苏贞昌_x 不_h 就_d 变_v 冲_k 公_k （_w 台语_nz ）_w ？_w  _w \n_w  _w \n_w  _w 柯文哲_nr 分享_v 完_h 这段_mq 故事_n 后_f 不_h 忘_v 再次_d 提醒_v 司仪_n ，_w 以后_f 就_d 不要_v 再_d 用_v 恭请_v 这种_mq 封建_k 时代_n 的_u 语言_n 。_w ","publishedTime":"2015-03-30 09:24:00","source":"台海网","appId":"recommend","docType":"doc","other":"source\u003dspider|!|channel\u003dnews|!|tags\u003d凤凰网资讯-台湾-台湾时政|!|imgNum\u003d0","features":["台湾","c","1.0","时政","c","-1.0","台湾时政","cn","1.0","国民党","et","0.1","苏贞昌","et","0.1","民进党","et","0.1","蔡英文","et","0.1","柯文哲","et","1.0","台北市府","et","0.1","吴伯雄","et","0.1","市长","x","-0.5","中国时报","x","-0.1","卫生局长","x","-0.1","卫生局","x","-0.1","黄世杰","nr","-0.1","台语","nz","-0.1","讲义","n","-0.1","台海网","n","-0.1","时代","n","-0.1","司仪","n","-0.5","语言","n","-0.5","伯公","n","-0.1","首长","n","-0.1"]}	标题,纠正,恭请,柯文哲,语言,台海网,中国时报,台北市府,举办,首长,司仪,恭请,市长,柯文哲,语言,哄堂大笑,卫生局,讲义,整理,语录,入列,卫生局长,黄世杰,市长,核心,价值,整理,市长,纯属,参考,牢记,盲从,柯文哲,透露,参加,国民党,会议,喜欢,吴伯雄,称为,伯公,民进党,主席,蔡英文,提起,还好,民进党,文化,苏贞昌,台语,柯文哲,分享,故事,提醒,司仪,恭请,语言	原标题：纠正恭请柯文哲：别用封建语言台海网3月30日讯据中国时报报道，台北市府29日举办首长领航营，司仪刚宣布恭请市长致词后，旋遭柯文哲打脸不要再用这种封建时代的语言，引起现场哄堂大笑。但卫生局在讲义中整理40条柯语录，连奇怪耶，听起来怪怪的都入列，卫生局长黄世杰澄清，市长的核心价值很重要，才会整理市长说过的重点，纯属参考不用牢记或盲从。柯文哲透露，有次去参加国民党一个会议，他们很喜欢互称什么公、什么公，像吴伯雄就被称为伯公。因此有次遇到民进党主席蔡英文，自己就和她提起这件事，并说还好民进党没有这种文化，不然像苏贞昌不就变冲公（台语）？柯文哲分享完这段故事后不忘再次提醒司仪，以后就不要再用恭请这种封建时代的语言。	时政',
        u'{"ID":"189","url":"http://digi.ifeng.com/a/20150330/41028905_0.shtml","title":"Apple Watch特别对待：无大客户优惠、无以旧换新","splitTitle":"Apple Watch_x 特别_k 对待_v ：_w 无_h 大_k 客户_n 优惠_k 、_w 无以_d 旧_k 换_v 新_k ","splitContent":"  _w 凤凰_k 数码_mq 讯_k  _w 3月_t 30日_t 消息_n ，_w 消息_n 人士_n 透露_v ，_w 苹果_k 为_v Mac_x ，_w iPhone_x ，_w iPad_x 提供_v 的_u 特别_d 企业_n 客户_n 定价_k ，_w 补贴_k ，_w 优惠_k 等_u ，_w 在_p Apple Watch_x 上市_v 初期_n 将_d 不_h 适用_a ，_w 不过_h 企业_n 客户_n 还是_c 可以_v 和_c Apple_x  _w Store_nx 里_h ，_w 专门_k 为_v 企业_n 大_k 客户服务_x 的_u 团队_n 联络_v ，_w 购买_v Apple Watch_x 。_w 现在_t 一些_n 企业_n 客户_n 已经_d 表示_k ，_w 有_v 兴趣_n 批量_k 购买_v Apple Watch_x 精英_h 不锈钢_x 版_k ，_w 富豪_k 18K_mq 金_e 版_k ，_w 因为_c 这_r 两_q 款_n 型号_n 的_u 定价_v 和_c 礼品性_n 非常_k 吸引_v 人_h 。_w  _w \n_w  _w \n_w        _w http://y3.ifengimg.com/haina/2015_14/6fe170b16fe9545.png    _w \n_w  _w \n_w   _w Apple Watch_x 特别_k 对待_v ：_w 无_h 大_k 客户_n 优惠_k 、_w 无以_d 旧_k 换_v 新_k   _w \n_w  _w \n_w  _w 另外_c ，_w 用户_n 参加_v iPhone_x ，_w iPad_x 等_u 苹果_k 的_u 以_p 旧_k 换_v 新_d 活动_v ，_w 获得_v 的_u Apple_x  _w Store_nx 代金_nz 券_n 将_d 不_h 能_v 用于_v 购买_v Apple Watch_x 。_w 所以_c ，_w 如果_c 用户_n 已经_d 有_v 新_n 的_u iPhone_x ，_w 旧_k 的_u iPhone_x 不_h 能_v 以_p 旧_k 换_v 新_n 买_k Apple Watch_x 。_w 但是_c 有_v 一个_mq 例外_k ，_w 如果_c 购买_v 新_n 的_u iPhone_x 以后_f ，_w 代金_nz 券_n 还有_v 余额_n ，_w 则_k 可以_v 用来_v 购买_v Apple Watch_x 。_w ","publishedTime":"2015-03-30 10:54:00","source":"凤凰数码","appId":"recommend","docType":"docpic","other":"source\u003dspider|!|channel\u003ddigi|!|tags\u003d凤凰网数码-极客-新品|!|imgNum\u003d1","features":["科技","c","-1.0","数码","c","1.0","极客","sc","1.0","苹果","cn","0.5","极客新品","cn","1.0","apple watch","et","1.0","ipad","et","0.1","mac","et","0.1","iphone","et","0.5","Apple","x","-0.1","客户服务","x","-0.1","不锈钢","x","-0.1","代金","nz","-0.1","客户","n","-0.5","礼品性","n","-0.1","消息","n","-0.1","用户","n","-0.1","一些","n","-0.1","余额","n","-0.1","团队","n","-0.1","型号","n","-0.1","人士","n","-0.1","初期","n","-0.1","兴趣","n","-0.1","Store","nx","-0.1","优惠","k","-0.5","补贴","k","-0.1","定价","k","-0.1","富豪","k","-0.1","凤凰","k","-0.1"]}	消息,消息,人士,透露,mac,iphone,ipad,提供,企业,客户,watch,上市,企业,客户,apple,企业,客户服务,团队,联络,购买,watch,企业,客户,兴趣,购买,watch,不锈钢,型号,定价,礼品性,吸引,watch,客户,换,用户,参加,iphone,ipad,换,活动,apple,代金,购买,watch,用户,iphone,iphone,换,watch,购买,iphone,代金,余额,购买,watch	凤凰数码讯3月30日消息，消息人士透露，苹果为Mac，iPhone，iPad提供的特别企业客户定价，补贴，优惠等，在AppleWatch上市初期将不适用，不过企业客户还是可以和AppleStore里，专门为企业大客户服务的团队联络，购买AppleWatch。现在一些企业客户已经表示，有兴趣批量购买AppleWatch精英不锈钢版，富豪18K金版，因为这两款型号的定价和礼品性非常吸引人。/haina/2015AppleWatch特别对待：无大客户优惠、无以旧换新另外，用户参加iPhone，iPad等苹果的以旧换新活动，获得的AppleStore代金券将不能用于购买AppleWatch。所以，如果用户已经有新的iPhone，旧的iPhone不能以旧换新买AppleWatch。但是有一个例外，如果购买新的iPhone以后，代金券还有余额，则可以用来购买AppleWatch。	科技']
    feature_mat = mainClassifier.data_to_feature(raw_document)
    # 只返回单分类
    classify_result = mainClassifier.classify_documents(feature_mat)
    print classify_result
    # 需要返回的类别数量
    k = 3
    # # 返回多个分类和其概率
    classify_results = mainClassifier.classify_documents_top_k(feature_mat, k)
    print classify_results


def main3():
    main_classifier = MainClassifier()
    for cur_classiier in ClassifierConfig.train_data_claasifiers:
        ClassifierConfig.cur_single_model = cur_classiier
        main_classifier.set_model()
        print ClassifierConfig.cur_single_model
        results = main_classifier.classify_documents_from_file(FilePathConfig.raw_news_path)
        Util.save_object_into_pkl(results,
                                  FilePathConfig.file_root_path + ClassifierConfig.cur_single_model + "-raw_results.txt")


def main4():
    x = FilePathConfig.file_root_path + "svm-raw_results.txt"
    x = Util.load_object_from_pkl(x)
    print x
    print len(x)


if __name__ == '__main__':
    main4()
