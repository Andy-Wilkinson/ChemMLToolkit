"""
This type stub file was generated by pyright.
"""

""" a set of functions for interacting with databases

 When possible, it's probably preferable to use a _DbConnection.DbConnect_ object

"""
def GetColumns(dBase, table, fieldString, user=..., password=..., join=..., cn=...): # -> List[Any]:
    """ gets a set of data from a table

      **Arguments**

       - dBase: database name

       - table: table name

       - fieldString: a string with the names of the fields to be extracted,
          this should be a comma delimited list

       - user and  password:

       - join: a join clause (omit the verb 'join')


      **Returns**

       - a list of the data

    """
    ...

def GetData(dBase, table, fieldString=..., whereString=..., user=..., password=..., removeDups=..., join=..., forceList=..., transform=..., randomAccess=..., extras=..., cn=...): # -> List[Any] | RandomAccessDbResultSet | DbResultSet | None:
    """ a more flexible method to get a set of data from a table

      **Arguments**

       - fields: a string with the names of the fields to be extracted,
            this should be a comma delimited list

       - where: the SQL where clause to be used with the DB query

       - removeDups indicates the column which should be used to screen
          out duplicates.  Only the first appearance of a duplicate will
          be left in the dataset.

      **Returns**

        - a list of the data


      **Notes**

        - EFF: this isn't particularly efficient

    """
    ...

def DatabaseToText(dBase, table, fields=..., join=..., where=..., user=..., password=..., delim=..., cn=...): # -> str:
    """ Pulls the contents of a database and makes a deliminted text file from them

      **Arguments**
        - dBase: the name of the DB file to be used

        - table: the name of the table to query

        - fields: the fields to select with the SQL query

        - join: the join clause of the SQL query
          (e.g. 'join foo on foo.bar=base.bar')

        - where: the where clause of the SQL query
          (e.g. 'where foo = 2' or 'where bar > 17.6')

        - user: the username for DB access

        - password: the password to be used for DB access

      **Returns**

        - the CSV data (as text)

    """
    ...

def TypeFinder(data, nRows, nCols, nullMarker=...):
    """

      finds the types of the columns in _data_

      if nullMarker is not None, elements of the data table which are
        equal to nullMarker will not count towards setting the type of
        their columns.

    """
    ...

def GetTypeStrings(colHeadings, colTypes, keyCol=...): # -> list[Unknown]:
    """  returns a list of SQL type strings
    """
    ...

def TextFileToDatabase(dBase, table, inF, delim=..., user=..., password=..., maxColLabelLen=..., keyCol=..., nullMarker=...): # -> None:
    """loads the contents of the text file into a database.

      **Arguments**

        - dBase: the name of the DB to use

        - table: the name of the table to create/overwrite

        - inF: the file like object from which the data should
          be pulled (must support readline())

        - delim: the delimiter used to separate fields

        - user: the user name to use in connecting to the DB

        - password: the password to use in connecting to the DB

        - maxColLabelLen: the maximum length a column label should be
          allowed to have (truncation otherwise)

        - keyCol: the column to be used as an index for the db

      **Notes**

        - if _table_ already exists, it is destroyed before we write
          the new data

        - we assume that the first row of the file contains the column names

    """
    ...

def DatabaseToDatabase(fromDb, fromTbl, toDb, toTbl, fields=..., join=..., where=..., user=..., password=..., keyCol=..., nullMarker=...): # -> None:
    """

     FIX: at the moment this is a hack

    """
    ...

if __name__ == '__main__':
    sio = ...
    dirLoc = ...
