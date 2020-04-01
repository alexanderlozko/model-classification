import requests as req
from bs4 import BeautifulSoup
import re


class GetOffers:
    """
    Offers parsing
    """

    @staticmethod
    def get_offers_1(link=None):
        """
        Get offers from 1th site

        :param link: site with offers
        :return: list of offers
        """

        print('Собираем предложения по улучшению работы с сайта stbank.by')
        offers_list = list()
        for i in range(1, 5):
           resp = req.get(link + str(i))
           soup = BeautifulSoup(resp.text, 'lxml')
           offers = soup.find_all('div', 'idea-post-content')
           for off in offers:
               offer = re.sub(r'\s+', ' ', (off.text))
               category = 'Offer'
               offers_list.append([offer, category])

        return offers_list

    @staticmethod
    def get_offers_2(link=None):
        """
        Get offers from 2d site

        :param link: site with offers
        :return: list of offers
        """

        print('Собираем предложения по улучшению работы с сайта kapital.kz...')
        offers_list = list()
        resp = req.get(link)
        soup = BeautifulSoup(resp.text, 'lxml')
        offers = soup.find_all('p')
        for of in offers[3:42]:
            of = of.text
            category = 'Offer'
            offers_list.append([of, category])

        return offers_list

    @staticmethod
    def get_offers_3(link=None):
        """
        Get offers from 3th site

        :param link: site with offers
        :return: list of offers
        """

        print('Собираем предложения по улучшению работы с сайта mtp-global.com...')
        offers_list = list()
        resp = req.get(link)
        soup = BeautifulSoup(resp.text, 'lxml')
        offers = soup.find_all('p')
        category = 'Offer'
        for of in offers[25:85]:
            of = of.text
            if of != '\n\n' and of != '' and of != '\n\t\xa0\n' and of != '\n\xa0 \xa0\n':
                offers_list.append([of, category])

        small_offers = soup.find_all('h2')
        for of in small_offers:
            of = of.text
            offers_list.append([of, category])

        return offers_list
