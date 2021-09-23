from django.db import models
from django.contrib.auth.models import User # add user portfolio page
from django.dispatch import receiver # svae file
from django.db.models.signals import post_save # save file
from . import storage
from bson.objectid import ObjectId
# Create your models here.
class Profile(models.Model):   
	user = models.OneToOneField(User, on_delete=models.CASCADE)

	@receiver(post_save, sender=User) #add this to save file
	def create_user_profile(sender, instance, created, **kwargs):
		if created:
			Profile.objects.create(user=instance)

	@receiver(post_save, sender=User) #add this to save file
	def save_user_profile(sender, instance, **kwargs):
		instance.profile.save()

	
class UploadFile(models.Model):
	title = models.CharField(max_length=50)
	file = models.FileField(upload_to='./filesupload', storage=storage.FieldStorage())
	dateTimeOfUpload = models.DateTimeField(auto_now = True)

class Contact(models.Model):
	email = models.EmailField()
	subject = models.CharField(max_length=255)
	message = models.TextField()

	def __str__(self):
		return self.email

#class File(models.Model):
	#date_time= models.DateTimeField()
	#Average_fuel_consumption = models.FloatField()
	#Average_speed = models.FloatField()
	#Distance_travelled = models.FloatField()
	#Engine_RPM = models.FloatField()
	#Fuel_used = models.FloatField()
	#Vehicle_speed = models.FloatField()

class File(models.Model):
	time = models.DateTimeField()
	Average_speed = models.FloatField()
	Distance_travelled = models.FloatField()
	Engine_RPM = models.FloatField()
	Fuel_used = models.FloatField()
	Vehicle_speed =  models.FloatField()
	Driver_Type = models.IntegerField()
	Insurance_Fee = models.IntegerField()



