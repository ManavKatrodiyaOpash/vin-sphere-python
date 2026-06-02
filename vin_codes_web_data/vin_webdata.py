import requests
from bs4 import BeautifulSoup

url_toyota = "https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Toyota/VIN_Codes"
url_nissan = "https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Nissan/VIN_Codes"
url_hyundai = "https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Hyundai/VIN_Codes"
url_honda = "https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Honda/VIN_Codes"

def get_vin_data(url, filename):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract all visible text
        text = soup.get_text(separator="\n")

        # Clean empty lines
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        clean_text = "\n".join(lines)

        with open(f"{filename}_vin_codes.txt", "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"Data saved to {filename}_vin_codes.txt")

    else:
        print("Failed:", response.status_code)

if __name__ == "__main__":
    get_vin_data(url_toyota, "toyota")
    get_vin_data(url_nissan, "nissan")
    get_vin_data(url_hyundai, "hyundai")
    get_vin_data(url_honda, "honda")