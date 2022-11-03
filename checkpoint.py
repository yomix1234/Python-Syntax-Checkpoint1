#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = int(input("Input an integer :n "))
n1 = int( "%s" % a )
n2 = int( "%s%s" % (a,a) )
n3 = int( "%s%s%s" % (a,a,a) )
print (n1+n2+n3)


# In[2]:


# Python program to print Even Numbers in a List

# list of numbers
list1 = [10, 21, 4, 45, 66, 93]

# iterating each number in list
for num in list1:

	# checking condition
	if num % 2 == 0:
		print(num, end=" ")


# In[4]:


nl=[]
for x in range(1500, 2701):
    if (x%7==0) and (x%5==0):
        nl.append(str(x))
print (','.join(nl))


# In[5]:


# Python program to find the factorial of a number provided by the user.

# change the value for a different result
num = 8

# To take input from the user
#num = int(input("Enter a number: "))

factorial = 1

# check if the number is negative, positive or zero
if num < 0:
   print("Sorry, factorial does not exist for negative numbers")
elif num == 0:
   print("The factorial of 0 is 1")
else:
   for i in range(1,num + 1):
       factorial = factorial*i
   print("The factorial of",num,"is",factorial)


# In[6]:


c = input()

temp = ""

for i in range(len(c)):

    if i%2==0:

        temp = temp+c[i]


# In[ ]:




