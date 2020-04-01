from bs4 import BeautifulSoup
import requests as req
import lxml


class ReviewsURLScraper:
    """
    Parsing positive, negative comments, hotlines from minfin.com.ua and positive comments from about.pumb.ua
    """

    @staticmethod
    def get_reviews_url(banks_url=None, page=None):
        """
        Get list of top 20 banks list from minfin.com.ua

        :param banks_url: link of main page
        :param page: string
        :return: list of banks links
        """

        reviews = list()
        resp = req.get(banks_url+'/banks/top/')
        soup = BeautifulSoup(resp.text, 'lxml')
        top_20_banks = soup.find_all('tr')[3:23]
        for bank in top_20_banks:
            link = str(bank.find('a')).split('\"')[1]
            reviews.append(banks_url + link + page)

        return reviews

    @staticmethod
    def get_pos_and_neg_reviews(reviews_url=None):
        """
        Parsing list of positive and negative comments

        :param reviews_url: list of url
        :return: list positive and negative comments
        """

        print('Собираем позитивные и негативные отзывы...')
        reviews_list = list()
        for url in reviews_url:
            for i in range(1, 6):
                resp = req.get(url+str(i))
                soup = BeautifulSoup(resp.text, 'lxml')
                reviews1 = soup.find_all('div', 'comment')
                for review in reviews1:
                    rating = int(len(review.find_all('div', 'mfb-stars mfb-stars--fill')))
                    if rating <= 3:
                        intonation = 'Negative'
                    else:
                        intonation = 'Positive'
                    text = review.find('div', 'text').text
                    reviews_data = [text, intonation]
                    reviews_list.append(reviews_data)

        return reviews_list

    @staticmethod
    def get_positive(pos_url=None):
        """
        Parsing list of positive comments

        :param pos_url: only positive comments link
        :return: list of positive comments
        """

        print('Добавляем только позитивные отзывы с about.pumb.ua...')
        positive_list = list()
        resp = req.get(pos_url)
        soup = BeautifulSoup(resp.text, 'lxml')
        positive = soup.find_all('div', 'txt-block')
        for pos in positive:
            pos_reviews = pos.find_all('p')
            for pos_re in pos_reviews[0::3]:
                intonation = 'Positive'
                positive_list.append([pos_re.text, intonation])

        return positive_list

    @staticmethod
    def get_hotlines(hotlines_url=None):
        """
        Parsing list of hotlines

        :param hotlines_url: list of link
        :return: list of hotlines
        """

        print('Собираем запросы на внутренние департаменты...')
        hotlines_list = list()
        for url in hotlines_url:
            resp = req.get(url)
            soup = BeautifulSoup(resp.text, 'lxml')
            hotline = soup.find_all('div', 'comment')
            for line in hotline:
                question = line.find('div', 'text').text
                intonation = 'Hotline'
                hotline_data = [question, intonation]
                hotlines_list.append(hotline_data)

        return hotlines_list
