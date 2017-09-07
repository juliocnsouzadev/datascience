
# coding: utf-8

# In[8]:


class Profile():
    def __init__(self, name, phone, company):
        self.name = name
        self.phone = phone
        self.company =company
    
    def to_string(self):
        return 'Name: ' + self.name + ', Phone: ' + self.phone + ', Company: ' + self.company
        

