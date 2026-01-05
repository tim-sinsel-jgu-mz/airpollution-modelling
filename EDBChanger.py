import csv
import xml.etree.ElementTree as ET
from datetime import datetime
import os

def update_emission_database(csv_filepath, edb_filepath, output_filepath):
    """
    Updates an ENVI-MET database file (.edb) with emission values from a CSV file.

    Args:
        csv_filepath (str): The absolute path to the input CSV file.
        edb_filepath (str): The absolute path to the input EDB database file.
        output_filepath (str): The absolute path for the updated EDB output file.
    """
    print(f"Reading emission data from: {csv_filepath}")
    
    # --- 1. Read Emission Data from CSV into a Dictionary ---
    # The dictionary will use the street ID as a key for quick lookups.
    emission_data = {}
    try:
        # Open the CSV file, specifying the semicolon delimiter and 'utf-8-sig' encoding
        # The 'utf-8-sig' encoding handles the Byte Order Mark (BOM) that Excel can add.
        with open(csv_filepath, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                # Use .strip() to remove any leading/trailing whitespace from the ID
                street_id = row['Street'].strip() 
                # For each value: replace comma with period, convert to float,
                # then format to 5 decimal places as a string.
                emission_data[street_id] = {
                    'PM2.5': f"{float(row['PM2.5'].replace(',', '.')):.5f}",
                    'PM10':  f"{float(row['PM10'].replace(',', '.')):.5f}",
                    'NO':    f"{float(row['NO'].replace(',', '.')):.5f}",
                    'NO2':   f"{float(row['NO2'].replace(',', '.')):.5f}"
                }
        print(f"Successfully loaded data for {len(emission_data)} streets.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        print("Please make sure the script is in the same folder as the CSV file.")
        return
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV file: {e}")
        print("Please ensure the CSV has the headers: 'Street', 'PM2.5', 'PM10', 'NO', 'NO2'")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    print(f"Reading ENVI-MET database from: {edb_filepath}")

    # --- 2. Parse the EDB XML File ---
    try:
        tree = ET.parse(edb_filepath)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: The file '{edb_filepath}' was not found.")
        print("Please make sure the script is in the same folder as the EDB file.")
        return
    except ET.ParseError as e:
        print(f"Error parsing the EDB file. It might not be a valid XML format. Details: {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading the EDB file: {e}")
        return

    # --- 3. Iterate Through Sources and Update Emission Profiles ---
    updated_sources_count = 0
    for source in root.findall('SOURCE'):
        source_id_element = source.find('ID')
        if source_id_element is not None:
            # Clean up the ID from the XML file to match the CSV key
            source_id = source_id_element.text.strip()
            
            if source_id in emission_data:
                print(f"Updating source: {source_id}...")
                data = emission_data[source_id]

                # Create the 24-hour emission profile string for each emitter.
                # It repeats the same value 24 times, separated by commas,
                # with a leading and trailing space as requested.
                pm25_profile = f" {','.join([data['PM2.5']] * 24)} "
                pm10_profile = f" {','.join([data['PM10']] * 24)} "
                no_profile = f" {','.join([data['NO']] * 24)} "
                no2_profile = f" {','.join([data['NO2']] * 24)} "

                # Find the specific emission profile tags and update their text
                source.find('Emissionprofile_PM25').text = pm25_profile
                source.find('Emissionprofile_PM10').text = pm10_profile
                source.find('Emissionprofile_NO').text = no_profile
                source.find('Emissionprofile_NO2').text = no2_profile
                
                updated_sources_count += 1
            else:
                print(f"Warning: No data found in CSV for source ID: {source_id}")

    # --- 4. Update the Revision Date in the Header ---
    revision_date_element = root.find('Header/revisiondate')
    if revision_date_element is not None:
        now = datetime.now()
        # Format the date to match the original file's format
        revision_date_element.text = now.strftime('%d.%m.%Y %H:%M:%S')
        print(f"Updated revision date to: {revision_date_element.text}")

    # --- 5. Write the Updated Content to a New File ---
    # It's safer to write to a new file than to overwrite the original.
    try:
        tree.write(output_filepath, encoding='utf-8', xml_declaration=False)
        print("-" * 30)
        print(f"Successfully updated {updated_sources_count} sources.")
        print(f"Output file saved as: {output_filepath}")
        print("-" * 30)
    except Exception as e:
        print(f"An error occurred while writing the updated EDB file: {e}")


if __name__ == '__main__':
    # --- Get the absolute path of the directory where the script is located ---
    # This makes the script runnable from any location, as it will always
    # look for the data files in its own directory.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for interactive environments where __file__ is not defined
        script_dir = os.getcwd()

    # Define the filenames, joining them with the script's directory path
    csv_file = os.path.join(script_dir, 'Sources_Calc.csv')
    edb_file = os.path.join(script_dir, 'projectdatabase.edb')
    output_file = os.path.join(script_dir, 'projectdatabase_updated.edb')
    
    # Run the update function
    update_emission_database(csv_file, edb_file, output_file)

