"""
This type stub file was generated by pyright.
"""

""" defines class _DbConnect_, for abstracting connections to databases

"""
class DbError(RuntimeError):
  ...


class DbConnect:
  """  This class is intended to abstract away many of the details of
    interacting with databases.

    It includes some GUI functionality

  """
  def __init__(self, dbName=..., tableName=..., user=..., password=...) -> None:
    """ Constructor

      **Arguments**  (all optional)

        - dbName: the name of the DB file to be used

        - tableName: the name of the table to be used

        - user: the username for DB access

        - password: the password to be used for DB access

    """
    ...
  
  def GetTableNames(self, includeViews=...): # -> list[str]:
    """ gets a list of tables available in a database

      **Arguments**

      - includeViews: if this is non-null, the views in the db will
        also be returned

      **Returns**

          a list of table names

      **Notes**

       - this uses _DbInfo.GetTableNames_

    """
    ...
  
  def GetColumnNames(self, table=..., join=..., what=..., where=..., **kwargs): # -> list[str]:
    """ gets a list of columns available in the current table

      **Returns**

          a list of column names

      **Notes**

       - this uses _DbInfo.GetColumnNames_

    """
    ...
  
  def GetColumnNamesAndTypes(self, table=..., join=..., what=..., where=..., **kwargs): # -> list[Unknown]:
    """ gets a list of columns available in the current table along with their types

      **Returns**

          a list of 2-tuples containing:

            1) column name

            2) column type

      **Notes**

       - this uses _DbInfo.GetColumnNamesAndTypes_

    """
    ...
  
  def GetColumns(self, fields, table=..., join=..., **kwargs): # -> List[Any]:
    """ gets a set of data from a table

      **Arguments**

       - fields: a string with the names of the fields to be extracted,
         this should be a comma delimited list

      **Returns**

          a list of the data

      **Notes**

        - this uses _DbUtils.GetColumns_

    """
    ...
  
  def GetData(self, table=..., fields=..., where=..., removeDups=..., join=..., transform=..., randomAccess=..., **kwargs): # -> List[Any] | RandomAccessDbResultSet | DbResultSet | None:
    """ a more flexible method to get a set of data from a table

      **Arguments**

       - table: (optional) the table to use

       - fields: a string with the names of the fields to be extracted,
         this should be a comma delimited list

       - where: the SQL where clause to be used with the DB query

       - removeDups: indicates which column should be used to recognize
         duplicates in the data.  -1 for no duplicate removal.

      **Returns**

          a list of the data

      **Notes**

        - this uses _DbUtils.GetData_

    """
    ...
  
  def GetDataCount(self, table=..., where=..., join=..., **kwargs): # -> Any:
    """ returns a count of the number of results a query will return

      **Arguments**

       - table: (optional) the table to use

       - where: the SQL where clause to be used with the DB query

       - join: the SQL join clause to be used with the DB query


      **Returns**

          an int

      **Notes**

        - this uses _DbUtils.GetData_

    """
    ...
  
  def GetCursor(self): # -> Cursor:
    """ returns a cursor for direct manipulation of the DB
      only one cursor is available

    """
    ...
  
  def KillCursor(self): # -> None:
    """ closes the cursor

    """
    ...
  
  def AddTable(self, tableName, colString): # -> None:
    """ adds a table to the database

    **Arguments**

      - tableName: the name of the table to add

      - colString: a string containing column definitions

    **Notes**

      - if a table named _tableName_ already exists, it will be dropped

      - the sqlQuery for addition is: "create table %(tableName) (%(colString))"

    """
    ...
  
  def InsertData(self, tableName, vals): # -> None:
    """ inserts data into a table

    **Arguments**

      - tableName: the name of the table to manipulate

      - vals: a sequence with the values to be inserted

    """
    ...
  
  def InsertColumnData(self, tableName, columnName, value, where): # -> None:
    """ inserts data into a particular column of the table

    **Arguments**

      - tableName: the name of the table to manipulate

      - columnName: name of the column to update

      - value: the value to insert

      - where: a query yielding the row where the data should be inserted

    """
    ...
  
  def AddColumn(self, tableName, colName, colType): # -> None:
    """ adds a column to a table

    **Arguments**

      - tableName: the name of the table to manipulate

      - colName: name of the column to insert

      - colType: the type of the column to add

    """
    ...
  
  def Commit(self): # -> None:
    """ commits the current transaction

    """
    ...
  


