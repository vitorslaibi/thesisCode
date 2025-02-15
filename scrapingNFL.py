import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from random import uniform

def extract_table_data(table):
    """Extract data from table using BeautifulSoup"""
    # First find all header rows
    thead = table.find('thead')
    header_rows = thead.find_all('tr')
    
    # Get all column headers, including subheaders
    headers = []
    for th in header_rows[-1].find_all(['th', 'td']):
        headers.append(th.get_text(strip=True))
    
    # Extract data rows
    rows = []
    tbody = table.find('tbody')
    for tr in tbody.find_all('tr'):
        # Skip rows that are section headers
        if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
            continue
        
        # Extract row data
        row_data = []
        for td in tr.find_all(['td', 'th']):
            cell_value = td.get_text(strip=True)
            row_data.append(cell_value)
        
        # Only add rows that have data and aren't playoff games
        if row_data and len(row_data) == len(headers):
            if not any(x in row_data[0] for x in ['Week', 'Playoffs', 'Bye']):
                rows.append(row_data)
    
    return headers, rows

def scrape_nfl_game_log():
    teams = [
        'buf', 'mia', 'nwe', 'nyj',  # AFC East
        'cin', 'cle', 'pit', 'rav',  # AFC North (Baltimore Ravens is 'rav')
        'htx', 'clt', 'jax', 'oti',  # AFC South (Houston is 'htx', Tennessee Titans is 'oti')
        'den', 'kan', 'rai', 'sdg',  # AFC West (Raiders is 'rai', Chargers is 'sdg')
        'dal', 'nyg', 'phi', 'was',  # NFC East
        'chi', 'det', 'gnb', 'min',  # NFC North (Green Bay is 'gnb')
        'atl', 'car', 'nor', 'tam',  # NFC South (New Orleans is 'nor', Tampa Bay is 'tam')
        'crd', 'ram', 'sfo', 'sea'   # NFC West (Cardinals is 'crd')
    ]
    
    all_game_logs = []

    for team in teams:
        print(f"\nScraping data for {team.upper()}...")
        url = f'https://www.pro-football-reference.com/teams/{team}/2019.htm'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        time.sleep(uniform(1, 3))
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'games'})
            
            if table is None:
                print(f"No table found for {team.upper()}")
                continue

            # Extract headers and rows
            headers, rows = extract_table_data(table)
            
            print(f"Found {len(headers)} columns: {headers}")
            print(f"Found {len(rows)} rows")
            
            # Create DataFrame with all columns
            df = pd.DataFrame(rows, columns=headers)
            
            # Add team identifier
            df['Team'] = team.upper()
            
            # Remove rows with missing Week values
            df = df[df['Week'].notna()]
            
            # Convert score columns to numeric
            score_cols = ['Tm', 'Opp']
            for col in score_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate Result if not present
            if 'Tm' in df.columns and 'Opp' in df.columns:
                df['Result'] = df.apply(lambda row: 
                    'W' if row['Tm'] > row['Opp']
                    else 'L' if row['Tm'] < row['Opp']
                    else 'T', axis=1)
            
            all_game_logs.append(df)
            print(f"Successfully scraped data for {team.upper()}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error scraping data for {team.upper()}: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error for {team.upper()}: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            print(f"Available columns: {df.columns.tolist() if 'df' in locals() else 'No DataFrame created'}")
            continue

    # Combine all DataFrames
    if all_game_logs:
        try:
            # First, get a list of all columns across all dataframes
            all_columns = set()
            for df in all_game_logs:
                all_columns.update(df.columns)
            
            # Fill missing columns with NaN before concatenating
            for i, df in enumerate(all_game_logs):
                missing_cols = all_columns - set(df.columns)
                for col in missing_cols:
                    all_game_logs[i][col] = pd.NA
            
            all_game_logs_df = pd.concat(all_game_logs, ignore_index=True)
            
            # Convert date column to datetime
            if 'Date' in all_game_logs_df.columns:
                all_game_logs_df['Date'] = pd.to_datetime(all_game_logs_df['Date'])
            
            # Sort by date if possible
            if 'Date' in all_game_logs_df.columns:
                all_game_logs_df = all_game_logs_df.sort_values('Date')
            
            # Reset index
            all_game_logs_df = all_game_logs_df.reset_index(drop=True)
            
            # Ensure core columns exist
            core_columns = ['Week', 'Date', 'Team', 'Opp', 'Tm', 'Result']
            missing_core = [col for col in core_columns if col not in all_game_logs_df.columns]
            if missing_core:
                print(f"Warning: Missing core columns: {missing_core}")
            
            return all_game_logs_df
            
        except Exception as e:
            print(f"Error combining data: {str(e)}")
            return None
    else:
        print("No data was scraped.")
        return None

def main():
    print("Starting NFL game log scraping for 2019 season...")
    game_logs_df = scrape_nfl_game_log()
    
    if game_logs_df is not None:
        output_file = 'nfl_game_logs_2019.csv'
        game_logs_df.to_csv(output_file, index=False)
        print(f"\nData scraping complete. Data saved to {output_file}")
        print(f"Total games scraped: {len(game_logs_df)}")
        print("\nDataset overview:")
        print(game_logs_df.info())
        
        # Print sample of data
        print("\nFirst few rows of the dataset:")
        print(game_logs_df.head())
    else:
        print("No data to save.")

if __name__ == '__main__':
    main()