
-- Stored Procedures for NGramUtilities



-- Create NGram Table

CREATE PROCEDURE create_ngram_table(cardinality INT)

BEGIN

    SET @query = CONCAT('CREATE TABLE IF NOT EXISTS _', cardinality, '_gram (');

    SET @unique = '';

    SET @i = cardinality;

    WHILE @i > 0 DO

        SET @i = @i - 1;

        IF @i != 0 THEN

            SET @unique = CONCAT(@unique, 'word_', @i, ', ');

            SET @query = CONCAT(@query, 'word_', @i, ' TEXT, ');

        ELSE

            SET @unique = CONCAT(@unique, 'word');

            SET @query = CONCAT(@query, 'word TEXT, count INTEGER, UNIQUE(', @unique, ') );');

        END IF;

    END WHILE;

    PREPARE stmt FROM @query;

    EXECUTE stmt;

    DEALLOCATE PREPARE stmt;

END;



-- Delete NGram Table

CREATE PROCEDURE delete_ngram_table(cardinality INT)

BEGIN

    SET @query = CONCAT('DROP TABLE IF EXISTS _', cardinality, '_gram;');

    PREPARE stmt FROM @query;

    EXECUTE stmt;

    DEALLOCATE PREPARE stmt;

END;



-- Create Index

CREATE PROCEDURE create_index(cardinality INT)

BEGIN

    SET @i = cardinality;

    WHILE @i > 0 DO

        SET @i = @i - 1;

        IF @i != 0 THEN

            SET @query = CONCAT('CREATE INDEX idx_', cardinality, '_gram_', @i, ' ON _', cardinality, '_gram(word_', @i, ');');

            PREPARE stmt FROM @query;

            EXECUTE stmt;

            DEALLOCATE PREPARE stmt;

        END IF;

    END WHILE;

END;



-- Delete Index

CREATE PROCEDURE delete_index(cardinality INT)

BEGIN

    SET @i = cardinality;

    WHILE @i > 0 DO

        SET @i = @i - 1;

        IF @i != 0 THEN

            SET @query = CONCAT('DROP INDEX IF EXISTS idx_', cardinality, '_gram_', @i, ';');

            PREPARE stmt FROM @query;

            EXECUTE stmt;

            DEALLOCATE PREPARE stmt;

        END IF;

    END WHILE;

END;



-- Insert NGram

CREATE PROCEDURE insert_ngram(cardinality INT, ngram TEXT, count INT)

BEGIN

    SET @values = CONCAT('VALUES(', ngram, ', ', count, ')');

    SET @query = CONCAT('INSERT INTO _', cardinality, '_gram ', @values, ';');

    PREPARE stmt FROM @query;

    EXECUTE stmt;

    DEALLOCATE PREPARE stmt;

END;



-- Update NGram

CREATE PROCEDURE update_ngram(cardinality INT, ngram TEXT, count INT)

BEGIN

    SET @query = CONCAT('UPDATE _', cardinality, '_gram SET count = ', count, ' WHERE word = ', ngram, ';');

    PREPARE stmt FROM @query;

    EXECUTE stmt;

    DEALLOCATE PREPARE stmt;

END;



-- Remove NGram

CREATE PROCEDURE remove_ngram(cardinality INT, ngram TEXT)

BEGIN

    SET @query = CONCAT('DELETE FROM _', cardinality, '_gram WHERE word = ', ngram, ';');

    PREPARE stmt FROM @query;

    EXECUTE stmt;

    DEALLOCATE PREPARE stmt;

END;
