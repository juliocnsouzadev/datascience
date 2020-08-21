import message_builder
import sender
import template_loader

def test():
    file = 'newsletter-template.html'
    data = {
        "welcome_message" : "Welcome!",
        "site_address" : "www.google.com",
        "name" : "Joao das Couves",
        "company" : "Arvere Inc",
        "email" : "zeh_das_couves@gmail.com",
        "reply_adress" : "arvers.inc@gmail.com"
    }
    template_text = template_loader.get_template(file)
    html_message = template_loader.render_template(template_text, data)
    from_email = "hungrypy@gmail.com"
    the_message = message_builder.buildMessage(from_email, "Testsg","some plain text", html_message)
    email = data.get("email")
    list_emails = [email]
    sender.sendmail(list_emails,the_message)

test()