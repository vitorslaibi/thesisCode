import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scrape_game_log():
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
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve data for {team}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'tgl_basic'})
        
        if table is None:
            print(f"No table found for {team}")
            continue

        df = pd.read_html(str(table))[0]
        
        # Clean the DataFrame (remove multi-level headers if they exist)
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(0)
        
        df['Team'] = team  # Add a column to identify the team
        
        
        # Drop rows where the "Rk" column is not numeric
        df = df[pd.to_numeric(df['Rk'], errors='coerce').notna()]

        all_game_logs.append(df)

    # Combine all DataFrames into one
    if all_game_logs:
        all_game_logs_df = pd.concat(all_game_logs, ignore_index=True) 
        
        return all_game_logs_df
    else:
        print("No data was scraped.")
        return None

def main():
    game_logs_df = scrape_game_log()
    
    if game_logs_df is not None:
        game_logs_df.to_csv('nba_game_logs_2018_2019.csv', index=False)
        print("Data scraping complete. Data saved to nba_game_logs_2018_2019.csv")
    else:
        print("No data to save.")

if __name__ == '__main__':
    main()
