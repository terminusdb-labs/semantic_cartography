from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

OPENAI_KEY = os.environ.get("OPENAI_KEY")
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_KEY,
)

def get_embeddings(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   response = client.embeddings.create(input=texts, model=model)
   embeddings = [e.embedding for e in response.data]
   return embeddings

def dot(v1,v2):
    return sum([x*y for x,y in zip(v1,v2)])

def compare_phones_1():
    """ Least significant digit """
    phone1 = 'phone number: +353 85 136 8736'
    phone2 = 'phone number: +353 85 136 8735'
    print(f"Phone distance for {phone1} and {phone2}")
    [pv1, pv2] = get_embeddings([phone1, phone2])
    distance = dot(pv1, pv2)
    print(f"distance {distance}")

def compare_phones_2():
    """ Country code """
    phone1 = 'phone number: +353 85 136 8736'
    phone2 = 'phone number: +352 85 136 8736'
    print(f"Phone distance for {phone1} and {phone2}")
    [pv1, pv2] = get_embeddings([phone1, phone2])
    distance = dot(pv1, pv2)
    print(f"distance {distance}")

def compare_phones_3():
    """ Normalized comparison"""
    phone1 = 'phone number: 353851368736'
    phone2 = 'phone number: 353851368737'
    print(f"Phone distance for {phone1} and {phone2}")
    [pv1, pv2] = get_embeddings([phone1, phone2])
    distance = dot(pv1, pv2)
    print(f"distance {distance}")

def compare_vin_1():
    """ VIN comparison """
    vin1 = 'vehicle identification number: 1HGBH41JXMN109186'
    vin2 = 'vehicle identification number: 1HGBH41JXMN109187'
    print(f"VIN distance for {vin1} and {vin2}")
    [vv1, vv2] = get_embeddings([vin1, vin2])
    distance = dot(vv1, vv2)
    print(f"distance {distance}")

def compare_vin_2():
    """ VIN comparison """
    vin1 = 'vehicle identification number: 1HGBH41JXMN109186'
    vin2 = 'vehicle identification number: 1GNBH41JXMN109187'
    print(f"VIN distance for {vin1} and {vin2}")
    [vv1, vv2] = get_embeddings([vin1, vin2])
    distance = dot(vv1, vv2)
    print(f"distance {distance}")

def compare_dates_1():
    """ DATE comparison with day distance """
    date1 = 'date of birth is: 05-25-1977'
    date2 = 'date of birth is: 05-24-1977'
    print(f"DATE distance for {date1} and {date2}")
    [dv1, dv2] = get_embeddings([date1, date2])
    distance = dot(dv1, dv2)
    print(f"distance {distance}")

def compare_dates_2():
    """ DATE comparison with year distance """
    date1 = 'date of birth is: 05-24-1977'
    date2 = 'date of birth is: 05-24-1978'
    print(f"DATE distance for {date1} and {date2}")
    [dv1, dv2] = get_embeddings([date1, date2])
    distance = dot(dv1, dv2)
    print(f"distance {distance}")


if __name__ == "__main__":
    compare_phones_1()
    compare_phones_2()
    compare_phones_3()
    compare_vin_1()
    compare_vin_2()
    compare_dates_1()
    compare_dates_2()
