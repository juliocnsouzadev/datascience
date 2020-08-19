# coding: utf-8

from Profile import Profile

class ProfileVip(Profile):
    
    def get_credits(self):
        return super(ProfileVip, self).get_likes() * 10
    
    def to_string(self):
        return super(ProfileVip, self).to_string() + ', Credits: ', str(self.get_credits())
