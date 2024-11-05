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
In order to properly work you should first register to get an ECWMF account [here](https://accounts.ecmwf.int/auth/realms/ecmwf/login-actions/registration?client_id=cds&tab_id=uP8OQT6ER-E).
Once registered, follow the next steps to configure the credentials for the processor:

1. Go to your *HOME* or *user* folder. 
  > In Linux/MacOS this folder is at `/home/your_user_name`. In windows it is usually at `C:\Users\your_user_name`.

2. Login to the [Climate Data Store](https://cds.climate.copernicus.eu/) with your ECMWF username and password, 
then type the next URL address [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)

3. Create a blank ASCII file in your *HOME* folder named `.cdsapirc` (please note the dot `.` at the beginning of the string),
and paste the two lines that appear in the black box of the webpage.

![](./figures/cds_key.png)

  > These two lines have the following format:
  ```
  url: https://cds.climate.copernicus.eu/api
  key: UID:API-KEY
  ```
  where *UID* and *API-KEY* are your Copernicus Climate Data Store credentials.

4. Login to the [Atmospheric Data Store](https://ads.atmosphere.copernicus.eu/) with your ECMWF username and password, 
then type the next URL address [https://ads.atmosphere.copernicus.eu/how-to-api](https://ads.atmosphere.copernicus.eu/how-to-api)

5. Create a blank ASCII file in your *HOME* folder named `.adsapirc` (please note the dot `.` at the beginning of the string),
and paste the two lines that appear in the black box of the webpage.

  > These two lines have the following format:
  ```
   url: https://ads.atmosphere.copernicus.eu/api
   key: UID:API-KEY
  ```
  where *UID* and *API-KEY* are your Copernicus Atmospheric Climate Data Store credentials.

## Usage
An example for using the package can be found in the script `process_era5.py`


