from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()
import sys

OPENAI_KEY = os.environ.get("OPENAI_KEY")
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_KEY,
)

def get_embeddings(texts, model="text-embedding-3-small"): # text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   response = client.embeddings.create(input=texts, model=model)
   embeddings = [e.embedding for e in response.data]
   return embeddings

def dot(v1,v2):
    return sum([x*y for x,y in zip(v1,v2)])

def distance(v1,v2):
   return (dot(v1,v2)-1.0) / -2.0

def compare_phones_1():
    """ Least significant digit """
    phone1 = 'phone number: +353 85 136 8736'
    phone2 = 'phone number: +353 85 136 8735'
    print(f"Phone dist for {phone1} and {phone2}")
    [pv1, pv2] = get_embeddings([phone1, phone2])
    dist = distance(pv1, pv2)
    print(f"dist {dist}")

def compare_phones_2():
    """ Country code """
    phone1 = 'phone number: +353 85 136 8736'
    phone2 = 'phone number: +352 85 136 8736'
    print(f"Phone dist for {phone1} and {phone2}")
    [pv1, pv2] = get_embeddings([phone1, phone2])
    dist = distance(pv1, pv2)
    print(f"dist {dist}")

def compare_phones_3():
    """ Normalized comparison"""
    phone1 = 'phone number: 353851368736'
    phone2 = 'phone number: 353851368737'
    print(f"Phone dist for {phone1} and {phone2}")
    [pv1, pv2] = get_embeddings([phone1, phone2])
    dist = distance(pv1, pv2)
    print(f"dist {dist}")

def compare_phones_4():
    """ Normalized comparison with significant digit"""
    phone1 = 'phone number: 353851368736'
    phone2 = 'phone number: 352851368736'
    print(f"Phone dist for {phone1} and {phone2}")
    [pv1, pv2] = get_embeddings([phone1, phone2])
    dist = distance(pv1, pv2)
    print(f"dist {dist}")

def compare_vin_1():
    """ VIN comparison """
    vin1 = 'vehicle identification number: 1HGBH41JXMN109186'
    vin2 = 'vehicle identification number: 1HGBH41JXMN109187'
    print(f"VIN dist for {vin1} and {vin2}")
    [vv1, vv2] = get_embeddings([vin1, vin2])
    dist = distance(vv1, vv2)
    print(f"dist {dist}")

def compare_vin_2():
    """ VIN comparison """
    vin1 = 'vehicle identification number: 1HGBH41JXMN109186'
    vin2 = 'vehicle identification number: 1GNBH41JXMN109187'
    print(f"VIN dist for {vin1} and {vin2}")
    [vv1, vv2] = get_embeddings([vin1, vin2])
    dist = distance(vv1, vv2)
    print(f"dist {dist}")

def compare_vin_3():
   vin1 = """Make	Chevrolet
Model	Monte Carlo
Model year	2002
Body style	2 Doors Coupe
Engine type	3.4L V6 OHV 12V FWD
Fuel type	Gasoline
Transmission	Automatic
Manufactured in	Canada
Body type	Coupe
Number of doors	2
Number of seats	5
Displacement SI	3392
Displacement CID	207
Displacement Nominal	3.4
Engine type	V6
Engine head	OHV
Engine valves	12
Engine cylinders	6
Engine aspiration	Naturally
Compression ratio	9.5
Engine HorsePower	180
Engine KiloWatts	134
Automatic gearbox	4AT
Fuel type	Gasoline
MPG city	21
MPG highway	32
Driveline	FWD
Anti-Brake System	4-Wheel ABS
Front brake type	Disc
Rear brake type	Disc
Front suspension	Independent
Rear suspension	Independent
Front spring type	Coil
Rear spring type	Coil
Tire front	225/60R16
Tire rear	225/60R16"""
   vin2 = """Make	Chevrolet
Model	Monte Carlo
Model year	2000
Body style	2 Doors Coupe
Engine type	3.4L L6 OHV 12V FWD
Fuel type	Gasoline
Transmission	Automatic
Manufactured in	Canada
Body type	Coupe
Number of doors	2
Number of seats	5
Displacement SI	3392
Displacement CID	207
Displacement Nominal	3.4
Engine type	L6
Engine head	OHV
Engine valves	12
Engine cylinders	6
Engine aspiration	Naturally
Compression ratio	9.6
Engine HorsePower	180
Engine KiloWatts	134
Automatic gearbox	4AT
Fuel type	Gasoline
MPG city	20
MPG highway	32
Driveline	FWD
Anti-Brake System	4-Wheel ABS
Front brake type	Disc
Rear brake type	Disc
Front suspension	Independent
Rear suspension	Independent
Front spring type	Coil
Rear spring type	Coil
Tire front	225/60R16
Tire rear	225/60R16"""
   print("Expanded vin comparison")
   [vv1, vv2] = get_embeddings([vin1, vin2])
   dist = distance(vv1, vv2)
   print(f"dist {dist}")

def compare_vin_4():
   """ Vin comparison to description """
   vin1 = "vehicle identification number: 2G1WW12E029173400"
   vin2 = """Make	Chevrolet
Model	Monte Carlo
Model year	2000
Body style	2 Doors Coupe
Engine type	3.4L L6 OHV 12V FWD
Fuel type	Gasoline
Transmission	Automatic
Manufactured in	Canada
Body type	Coupe
Number of doors	2
Number of seats	5
Displacement SI	3392
Displacement CID	207
Displacement Nominal	3.4
Engine type	L6
Engine head	OHV
Engine valves	12
Engine cylinders	6
Engine aspiration	Naturally
Compression ratio	9.6
Engine HorsePower	180
Engine KiloWatts	134
Automatic gearbox	4AT
Fuel type	Gasoline
MPG city	20
MPG highway	32
Driveline	FWD
Anti-Brake System	4-Wheel ABS
Front brake type	Disc
Rear brake type	Disc
Front suspension	Independent
Rear suspension	Independent
Front spring type	Coil
Rear spring type	Coil
Tire front	225/60R16
Tire rear	225/60R16"""
   print("Expanded versus unexpanded vin comparison, identical")
   [vv1, vv2] = get_embeddings([vin1, vin2])
   dist = distance(vv1, vv2)
   print(f"dist {dist}")

def compare_vin_5():
   """ Vin comparison to description """
   vin1 = "vehicle identification number: 2T2BK1BA7CC121628"
   vin2 = """Make	Chevrolet
Model	Monte Carlo
Model year	2000
Body style	2 Doors Coupe
Engine type	3.4L L6 OHV 12V FWD
Fuel type	Gasoline
Transmission	Automatic
Manufactured in	Canada
Body type	Coupe
Number of doors	2
Number of seats	5
Displacement SI	3392
Displacement CID	207
Displacement Nominal	3.4
Engine type	L6
Engine head	OHV
Engine valves	12
Engine cylinders	6
Engine aspiration	Naturally
Compression ratio	9.6
Engine HorsePower	180
Engine KiloWatts	134
Automatic gearbox	4AT
Fuel type	Gasoline
MPG city	20
MPG highway	32
Driveline	FWD
Anti-Brake System	4-Wheel ABS
Front brake type	Disc
Rear brake type	Disc
Front suspension	Independent
Rear suspension	Independent
Front spring type	Coil
Rear spring type	Coil
Tire front	225/60R16
Tire rear	225/60R16"""
   print("Expanded versus unexpanded vin comparison, unrelated")
   [vv1, vv2] = get_embeddings([vin1, vin2])
   dist = distance(vv1, vv2)
   print(f"dist {dist}")

def compare_vin_6():
   """ Vin comparison to description """
   vin1 = """Make	Lexus
Model	RX 350
Model year	2012
Body style	4 Doors SUV
Engine type	3.5L V6 DOHC 24V AWD
Transmission	5-Speed Automatic
Manufactured in	Canada
Body type	SUV
Number of doors	4
Number of seats	5
Displacement SI	3458
Displacement CID	211
Displacement Nominal	3.5
Engine type	V6
Engine head	DOHC
Engine valves	24
Engine cylinders	6
Engine aspiration	Naturally
Engine HorsePower	275
Engine KiloWatts	205
Automatic gearbox	5AT
Fuel type	Gasoline
Driveline	AWD
Anti-Brake System	ABS
GVWR range	5001-6000"""
   vin2 = """Make	Chevrolet
Model	Monte Carlo
Model year	2000
Body style	2 Doors Coupe
Engine type	3.4L L6 OHV 12V FWD
Fuel type	Gasoline
Transmission	Automatic
Manufactured in	Canada
Body type	Coupe
Number of doors	2
Number of seats	5
Displacement SI	3392
Displacement CID	207
Displacement Nominal	3.4
Engine type	L6
Engine head	OHV
Engine valves	12
Engine cylinders	6
Engine aspiration	Naturally
Compression ratio	9.6
Engine HorsePower	180
Engine KiloWatts	134
Automatic gearbox	4AT
Fuel type	Gasoline
MPG city	20
MPG highway	32
Driveline	FWD
Anti-Brake System	4-Wheel ABS
Front brake type	Disc
Rear brake type	Disc
Front suspension	Independent
Rear suspension	Independent
Front spring type	Coil
Rear spring type	Coil
Tire front	225/60R16
Tire rear	225/60R16"""
   print("Expanded vin comparison, unrelated")
   [vv1, vv2] = get_embeddings([vin1, vin2])
   dist = distance(vv1, vv2)
   print(f"dist {dist}")

def compare_dates_1():
    """ DATE comparison with day dist """
    date1 = 'date of birth is: 05-25-1977'
    date2 = 'date of birth is: 05-24-1977'
    print(f"DATE dist for {date1} and {date2}")
    [dv1, dv2] = get_embeddings([date1, date2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_dates_2():
    """ DATE comparison with year dist """
    date1 = 'date of birth is: 05-24-1977'
    date2 = 'date of birth is: 05-24-1978'
    print(f"DATE dist for {date1} and {date2}")
    [dv1, dv2] = get_embeddings([date1, date2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_dates_3():
    """ DATE comparison with day dist in ISO """
    date1 = 'date of birth is: 25-05-1977'
    date2 = 'date of birth is: 24-05-1977'
    print(f"DATE dist for {date1} and {date2}")
    [dv1, dv2] = get_embeddings([date1, date2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_dates_4():
    """ DATE comparison with year dist in ISO """
    date1 = 'date of birth is: 25-05-1977'
    date2 = 'date of birth is: 24-05-1978'
    print(f"DATE dist for {date1} and {date2}")
    [dv1, dv2] = get_embeddings([date1, date2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_1():
    """  """
    name1 = 'His first name is Richard'
    name2 = 'His first name is Richaard'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_2():
    """  """
    name1 = 'His first name is richard'
    name2 = 'His first name is richaard'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_3():
    """  """
    name1 = 'His first name is richard'
    name2 = 'His first name is dick'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_4():
    """  """
    name1 = 'His first name is frank'
    name2 = 'His first name is francis'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_5():
    """  """
    name1 = 'His first name is mike'
    name2 = 'His first name is morris'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_6():
    """  """
    name1 = 'A person named frank t williams'
    name2 = 'A person named francis williams'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_7():
    """  """
    name1 = 'A person named frank williams'
    name2 = 'A person named francis williams'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_8():
    """  """
    name1 = 'A person named mike harper'
    name2 = 'A person named michael harper'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_9():
    """  """
    name1 = 'A person named george w bush jr'
    name2 = 'A person named george h w bush'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_10():
    """  """
    name1 = 'A person named gavin ellis mendel-gleason'
    name2 = 'A person named matthijs van otterdijk'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_11():
    """  """
    name1 = 'A person named Sasha'
    name2 = 'A person named Alexander'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def compare_names_12():
    """  """
    name1 = 'A person named Albert'
    name2 = 'A person named Alexander'
    print(f"Name dist for {name1} and {name2}")
    [dv1, dv2] = get_embeddings([name1, name2])
    print(dv1)
    dist = distance(dv1, dv2)
    print(f"dist {dist}")

def overlap():
    """  """
    record1 = 'There is a person named Jim. He lives at 33 Elm Street. He is 33 years old.'
    record2 = 'There is a person named Jim.'
    record3 = 'He lives at 33 Elm Street. He is 33 years old.'
    print(f"Overlap distance")
    [dv1, dv2, dv3] = get_embeddings([record1, record2, record3])
    dist1_2 = distance(dv1, dv2)
    dist1_3 = distance(dv1, dv3)
    dist2_3 = distance(dv2, dv3)
    print(f"dist 1,2 {dist1_2}")
    print(f"dist 1,3 {dist1_3}")
    print(f"dist 2,3 {dist2_3}")

def anchorage_accounting():
   record1 = 'Anchorage'
   record2 = 'Accounting'
   [dv1, dv2] = get_embeddings([record1, record2])
   dist = distance(dv1, dv2)
   print(f"dist {record1} {record2} = {dist}")

def anchorage_alaska():
   record1 = 'Anchorage'
   record2 = 'Alaska'
   [dv1, dv2] = get_embeddings([record1, record2])
   dist = distance(dv1, dv2)
   print(f"dist {record1} {record2} = {dist}")

def albany_new_york():
   record1 = 'Albany'
   record2 = 'New York'
   [dv1, dv2] = get_embeddings([record1, record2])
   dist = distance(dv1, dv2)
   print(f"dist {record1} {record2} = {dist}")

def albany_new_york():
   record1 = 'Albany'
   record2 = 'New York'
   [dv1, dv2] = get_embeddings([record1, record2])
   dist = distance(dv1, dv2)
   print(f"dist {record1} {record2} = {dist}")

def sciedam_new_york():
   record1 = 'Schiedam'
   record2 = 'New York'
   [dv1, dv2] = get_embeddings([record1, record2])
   dist = distance(dv1, dv2)
   print(f"dist {record1} {record2} = {dist}")

def sciedam_netherlands():
   record1 = 'Schiedam'
   record2 = 'Netherlands'
   [dv1, dv2] = get_embeddings([record1, record2])
   dist = distance(dv1, dv2)
   print(f"dist {record1} {record2} = {dist}")


def sciedam_netherlands_truth():
   record1 = 'Is Schiedam in the Netherlands'
   record2 = 'Is Schiedam in the Netherlands? No.'
   record3 = 'Is Schiedam in the Netherlands? Yes.'
   [dv1, dv2, dv3] = get_embeddings([record1, record2, record3])
   dist1_2 = distance(dv1, dv2)
   dist1_3 = distance(dv1, dv3)
   print(f"dist 1 2 {record1} {record2} = {dist1_2}")
   print(f"dist 1 3 {record1} {record3} = {dist1_3}")

def sciedam_netherlands_truth():
   record1 = 'Is Schiedam in New York?'
   record2 = 'Is Schiedam in New York? No.'
   record3 = 'Is Schiedam in New York? Yes.'
   [dv1, dv2, dv3] = get_embeddings([record1, record2, record3])
   dist1_2 = distance(dv1, dv2)
   dist1_3 = distance(dv1, dv3)
   print(f"dist 1 2 {record1} {record2} = {dist1_2}")
   print(f"dist 1 3 {record1} {record3} = {dist1_3}")

if __name__ == "__main__":
   sciedam_netherlands_truth()
   sys.exit(0)
   anchorage_alaska()
   anchorage_accounting()
   albany_new_york()
   sciedam_new_york()
   sciedam_netherlands()
   sys.exit(0)
   """
    compare_phones_1()
    compare_phones_2()
    compare_phones_3()
    compare_phones_4()
    compare_vin_1()
    compare_vin_2()
    compare_vin_3()
    compare_vin_4()
    compare_vin_5()
    compare_vin_6()
    compare_dates_1()
    compare_dates_2()
    compare_dates_3()
    compare_dates_4()
    compare_names_1()
    compare_names_2()
    compare_names_3()
    compare_names_4()
    compare_names_5()
    compare_names_6()
    compare_names_7()
    compare_names_8()
    compare_names_9()
    compare_names_10()
    compare_names_11()
    compare_names_12()
   """
   overlap()
