import pandas as pd


class ToxicComments:
    """
    Creating list of toxic comments from default file
    """

    @staticmethod
    def read_file(file_name=None):
        """
        Reading default file

        :param file_name: file with toxic comments
        :return: read file
        """

        fixed_df = pd.read_csv(file_name, delimiter=',',
                               names=['text', 'toxic'])

        return fixed_df

    @staticmethod
    def create_toxic_list(fixed_df=None):
        """
        Creating list of degree of toxicity

        :param fixed_df: read file
        :return: new list with toxic
        """

        new_toxic_list = list()
        for ls in fixed_df['toxic']:
            new_toxic_list.append(int(ls))

        return new_toxic_list

    @staticmethod
    def create_text_list(fixed_df=None):
        """
        Creating list of comment text

        :param fixed_df: read file
        :return: new list with text
        """

        new_text_list = list()
        for ls in fixed_df['text']:
            new_text_list.append(ls)

        return new_text_list

    @staticmethod
    def create_final_toxic(new_text_list=None, new_toxic_list=None):
        """
        Creating list of the most toxiс comments

        :param new_text_list: list with toxic comments
        :param new_toxic_list: list with toxic category
        :return: list of toxic comments
        """

        print('Cобираем хулиганские сообщения...')
        new_toxic = list()
        n = 0
        for comments in new_text_list:
            new_toxic.append([new_text_list[n], new_toxic_list[n]])
            n += 1
        toxic_list = new_toxic[:200]
        for ls in toxic_list:
            if ls[1] == 1:
                ls[1] = 'Hooligan'
            else:
                toxic_list.remove(ls)

        for ls in toxic_list:
            if ls[1] == 0 or ls[1] == 1:
                toxic_list.remove(ls)

        return toxic_list
