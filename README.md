# Meteo Utils
This Python repository contains methods that allow the automatic download and processing of ECMWF data relevant for evapotranspiration modelling.

## Installation
With `conda`, you can create a complete environment with all the required librarie typing
```
conda env create -f environment.yml
```
or update a current environment:
```
conda env update -n <environment> -f environment.yml
```

## Requirements
### Registration at the Copernicus climate and atmoshperic data stores
This plugin allows the automatic download and preprocessing of ERA5 meteorological data, provided by ECMWF.
In order to properly work you should first register to the Copernicus' [Climate Data Store](https://cds.climate.copernicus.eu/user/register)
and [Atmospheric Data Store](https://ads.atmosphere.copernicus.eu/user/register) systems.
Once registered, follow the next steps to configure the credentials for the QGIS ET processor:

1. Go to your *HOME* or *user* folder. 
  > In Linux/MacOS this folder is at `/home/your_user_name`. In windows it is usually at `C:\Users\your_user_name`.

2. Login to Sthe [Climate Data Store](https://cds.climate.copernicus.eu/user/login?) with your username and password, 
then type the next URL address [https://cds.climate.copernicus.eu/api-how-to/#install-the-cds-api-key](https://cds.climate.copernicus.eu/api-how-to/#install-the-cds-api-key)

3. Create a blank ASCII file in your *HOME* folder named `.cdsapirc` (please note the dot `.` at the beginning of the string),
and paste the two lines that appear in the black box of the webpage.

![](./figures/cds_key.png)

  > These two lines have the following format:
  ```
  url: https://cds.climate.copernicus.eu/api/v2
  key: UID:API-KEY
  ```
  where *UID* and *API-KEY* are your Copernicus Climate Data Store credentials.

4. Login to the [Atmospheric Data Store](https://ads.atmosphere.copernicus.eu/user/login?) with your username and password, 
then type the next URL address [https://ads.atmosphere.copernicus.eu/api-how-to/#install-the-cds-api-key](https://ads.atmosphere.copernicus.eu/api-how-to/#install-the-cds-api-key)

5. Create a blank ASCII file in your *HOME* folder named `.adsapirc` (please note the dot `.` at the beginning of the string),
and paste the two lines that appear in the black box of the webpage.

  > These two lines have the following format:
  ```
   url: https://ads.atmosphere.copernicus.eu/api/v2
   key: UID:API-KEY
  ```
  where *UID* and *API-KEY* are your Copernicus Atmospheric Climate Data Store credentials.

## Usage
An example for using the package can be found in the script `process_era5.py`


