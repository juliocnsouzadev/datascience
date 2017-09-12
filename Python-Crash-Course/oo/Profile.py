
# coding: utf-8

# In[8]:


class Profile:
    def __init__(self, name, phone, company):
        self.__name = name
        self.__phone = phone
        self.__company =company
        self.__likes = 0
    
    def to_string(self):
        return 'Name: ' + str(self.__name) + ', Phone: ' + str(self.__phone) + ', Company: ' + str(self.__company) + ', Likes: ' + str(self.__likes)
    
    def show_likes(self):
        return self.__likes
    
    def like(self):
        self.__likes += 1