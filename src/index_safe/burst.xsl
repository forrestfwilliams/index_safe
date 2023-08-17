<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:template match="/">
    <burst>
      <manifest>
        <xsl:copy-of select="document(files/@manifest)/*"/>
      </manifest>
      <metadata>
        <xsl:for-each select="/files/file">
          <xsl:variable name="root" select="name(document(@name)/*)"/>
            <xsl:element name="{$root}">
              <xsl:attribute name="source_filename">
                <xsl:value-of select="@label"/>
              </xsl:attribute>
              <xsl:copy-of select="document(@name)/*/adsHeader/swath"/>
              <xsl:copy-of select="document(@name)/*/adsHeader/polarisation"/>
              <content>
                <xsl:copy-of select="document(@name)/*/*"/>
              </content>
            </xsl:element>
        </xsl:for-each>
      </metadata>
    </burst>
  </xsl:template>
</xsl:stylesheet>
