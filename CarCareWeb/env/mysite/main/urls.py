from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = "main"   

urlpatterns = [
    path("", views.homepage, name="homepage"),
    path("register", views.register, name="register"), #add registeration
    path("login", views.login_request, name ="login"), # add login function
    path("logout", views.logout_request, name= "logout"), # add logout function
    path("user", views.userpage, name = "userpage"),  # add for userpage 
    path("upload",views.upload_file,name="upload" ),
    path("aboutus", views.about_us, name= "aboutus"), # add About us page
    path("data", views.data, name= "data"),
    path("data_analysis", views.data_analysis, name= "data_analysis"),
    path("data_prediction", views.data_prediction, name= "data_prediction"),
    path("contact_us", views.contact_us,name="contact_us" ),
]+ static(settings.MEDIA_URL,document_roots=settings.MEDIA_ROOT )
