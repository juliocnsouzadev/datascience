# coding: utf-8

class Profile(object):
    def __init__(self, name, phone, company):
        self.__name = name
        self.__phone = phone
        self.__company =company
        self.__likes = 0
    
    def to_string(self):
        return 'Name: ' + str(self.__name) + ', Phone: ' + str(self.__phone) + ', Company: ' + str(self.__company) + ', Likes: ' + str(self.__likes)
    
    def get_likes(self):
        return self.__likes
    
    def like(self):
        self.__likes += 1
    
    @classmethod
    def generate(clazz,file_name):
        reader = open(file_name, 'r')
        profiles = []
        for line in reader:
            values = line.replace('\n','').split(',')
            profiles.append(clazz(*values))
        reader.close()
        return profiles
           